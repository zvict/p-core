import torch
import torch.nn as nn
from torch import autocast
from torch.nn import functional as F
import tinycudann as tcnn
import math
import random
from .mlp import MLP, StackedLinearLayers
from .utils import activation_func, normalize_vector, get_tcnn_init_weights, Encoding, LayerNorm, rectify_points
from .sh import eval_sh, eval_sh_bases

# Import FlexAttention if available
try:
    from torch.nn.attention import flex_attention
    FLEX_ATTN_AVAILABLE = True
except ImportError:
    FLEX_ATTN_AVAILABLE = False
    print("Warning: FlexAttention not available. Falling back to standard attention.")


C0 = 0.28209479177387814


class MultiheadAttentionRaw(nn.Module):
    """
    Custom MultiheadAttention that returns raw attention logits (pre-softmax) and affined values.
    Based on PyTorch's MultiheadAttention implementation.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None):
        super(MultiheadAttentionRaw, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = nn.Parameter(torch.empty((embed_dim, embed_dim), device=device, dtype=dtype))
            self.k_proj_weight = nn.Parameter(torch.empty((embed_dim, self.kdim), device=device, dtype=dtype))
            self.v_proj_weight = nn.Parameter(torch.empty((embed_dim, self.vdim), device=device, dtype=dtype))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim), device=device, dtype=dtype))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim, device=device, dtype=dtype))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, device=device, dtype=dtype)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty((1, 1, embed_dim), device=device, dtype=dtype))
            self.bias_v = nn.Parameter(torch.empty((1, 1, embed_dim), device=device, dtype=dtype))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.q_proj_weight)
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def _scaled_dot_product_attention(self, q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, return_raw_logits=False):
        """
        Custom scaled dot product attention that returns raw logits and projected values.
        No softmax or attention weighted sum - just raw logits and projected values.
        """
        L, S = q.size(-2), k.size(-2)
        scale_factor = 1 / math.sqrt(q.size(-1))
        attn_bias = None

        if is_causal:
            temp_mask = torch.ones(L, S, dtype=torch.bool, device=q.device, requires_grad=False)
            temp_mask = temp_mask.tril(diagonal=0)
            attn_bias = torch.zeros(L, S, dtype=q.dtype, device=q.device)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias = attn_bias.unsqueeze(0)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.logical_not()
                attn_bias = attn_mask
            else:
                attn_bias = attn_mask

        # Compute raw attention logits
        raw_logits = q @ k.transpose(-2, -1) * scale_factor
        
        if attn_bias is not None:
            raw_logits = raw_logits + attn_bias
        
        # Return values as-is (they're already in multi-head format)
        # The forward method will handle reshaping back to original format
        projected_values = v
        
        # Return raw logits and projected values (no softmax, no attention weighted sum)
        return projected_values, None, raw_logits

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, 
                average_attn_weights=True, is_causal=False, return_raw_logits=False):
        """
        Args:
            query, key, value: map a query and a set of key-value pairs to an output.
            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. This is an binary mask. When the value is True,
                the corresponding value in the key will be ignored by the attention.
            need_weights: ignored (always returns None for attention weights).
            attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be
                broadcasted for all the batches while a 3D mask allows to specify a different mask for
                the entries of each batch.
            average_attn_weights: ignored (no attention weights returned).
            is_causal: If specified, applies a causal mask as attention mask, and ignores
                attn_mask for computing attention.
            return_raw_logits: ignored (always returns raw logits).

        Returns:
            attn_output: (L, N, E) projected values where L is the target sequence length, N is the batch size, E is the embedding dimension.
            attn_output_weights: Always None (no softmax applied).
            raw_logits: Raw attention logits (pre-softmax) of shape (N, num_heads, L, S) where S is the source sequence length.
        """
        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            query, key, value = query.transpose(1, 0), key.transpose(1, 0), value.transpose(1, 0)

        if not is_batched:
            # unsqueeze if the input is unbatched
            query = query.unsqueeze(1)
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.unsqueeze(0)

        # set up shape vars
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape
        assert embed_dim == self.embed_dim, f"was expecting embedding dimension of {self.embed_dim}, but got {embed_dim}"

        # compute in-projection
        if self._qkv_same_embed_dim:
            _b = self.in_proj_bias
            _start = 0
            _end = embed_dim
            _w = self.in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            # compute key-value
            _b = self.in_proj_bias
            _start = embed_dim
            _end = None
            _w = self.in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            k, v = F.linear(key, _w, _b).chunk(2, dim=-1)
        else:
            _b = self.in_proj_bias
            _start = 0
            _end = embed_dim
            _w = self.in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            _b = self.in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = self.in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = F.linear(key, _w, _b)

            _b = self.in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = self.in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = F.linear(value, _w, _b)

        # prep attention mask
        if attn_mask is not None:
            if attn_mask.dtype == torch.uint8:
                attn_mask = attn_mask.to(torch.bool)
            elif attn_mask.dtype == torch.bool:
                pass
            else:
                raise RuntimeError(f"attn_mask's dtype {attn_mask.dtype} is not supported")

        # prep key padding mask
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            key_padding_mask = key_padding_mask.to(torch.bool)

        # add bias along batch dimension (currently second)
        if self.bias_k is not None and self.bias_v is not None:
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))

        # reshape q, k, v for multihead attention and make em batch first
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(k.shape[0], bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(v.shape[0], bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # update source sequence length after adjustments
        src_len = k.size(1)

        # merge key padding and attention masks
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (bsz, src_len), \
                f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
            key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
                expand(-1, self.num_heads, -1, -1).reshape(bsz * self.num_heads, 1, src_len)
            if attn_mask is None:
                attn_mask = key_padding_mask
            elif attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.logical_or(key_padding_mask)
            else:
                attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

        # convert mask to float
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask

        # adjust dropout probability
        if not self.training:
            dropout_p = 0.0
        else:
            dropout_p = self.dropout

        # Calculate raw logits and projected values (no softmax, no attention weighted sum)
        projected_values, _, raw_logits = self._scaled_dot_product_attention(
            q, k, v, attn_mask, dropout_p, is_causal, return_raw_logits=True
        )
        
        # Reshape projected values back to original shape: (bsz * num_heads, src_len, head_dim) -> (src_len, bsz, embed_dim)
        # First transpose to (src_len, bsz * num_heads, head_dim), then reshape to (src_len, bsz, embed_dim)
        actual_src_len = projected_values.size(1)  # Use actual sequence length from the tensor
        projected_values = projected_values.transpose(0, 1).contiguous().view(actual_src_len, bsz, embed_dim)
        attn_output = F.linear(projected_values, self.out_proj.weight, self.out_proj.bias)

        # Reshape raw_logits from (bsz * num_heads, tgt_len, src_len) to (tgt_len, bsz, src_len)
        if raw_logits is not None:
            # raw_logits shape: (bsz * num_heads, tgt_len, actual_src_len)
            # Reshape to: (bsz, num_heads, tgt_len, actual_src_len)
            raw_logits = raw_logits.view(bsz, self.num_heads, tgt_len, actual_src_len)
            # Average across heads: (bsz, tgt_len, actual_src_len)
            raw_logits = raw_logits.mean(dim=1)
            # Transpose to: (tgt_len, bsz, actual_src_len)
            raw_logits = raw_logits.transpose(0, 1)

        if self.batch_first and is_batched:
            attn_output = attn_output.transpose(1, 0)
            if raw_logits is not None:
                raw_logits = raw_logits.transpose(0, 1)

        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
            if raw_logits is not None:
                raw_logits = raw_logits.squeeze(0)

        # Always return raw logits, no attention weights
        return attn_output, None, raw_logits


class Embedding(nn.Module):
    def __init__(self, input_dim, args, additional_dim=0):
        super(Embedding, self).__init__()
        self.args = args
        self.additional_dim = additional_dim
        
        self.encoder = Encoding(input_dim, args.encode_config, use_tcnn_encoder=False)
        encoder_output_dim = self.encoder.n_output_dims + additional_dim
        self.norm_in = LayerNorm(encoder_output_dim) if args.norm_in == "layernorm" else None
        self.norm_out = LayerNorm(args.output_dim) if args.norm_out == "layernorm" else None
        self.project = nn.Linear(args.output_dim, args.output_dim) if args.project else None
        if self.project is not None:
            nn.init.xavier_uniform_(self.project.weight)
        self.linear_layers = StackedLinearLayers(num_layers=args.n_hidden_layers,
                                                 dim_input=encoder_output_dim,
                                                 dim_output=args.output_dim,
                                                 dim_features=args.n_neurons,
                                                 nonlinearity=args.act,
                                                 output_add_nonlinearity=args.output_add_nonlinearity)
            
            
    def forward(self, x, additional_x=None):
        x = self.encoder(x)
        if self.additional_dim > 0 and additional_x is not None:
            x = torch.cat([x, additional_x], dim=-1)
        elif self.additional_dim > 0:
            raise ValueError("Additional dimension is > 0, but additional_x is None")
        if self.norm_in is not None:
            x = self.norm_in(x)
        x = self.linear_layers(x)
        if self.norm_out is not None:
            x = self.norm_out(x)
        if self.project is not None:
            x = self.project(x)
        return x      


class ProximitySelfAttention(nn.Module):
    def __init__(self, args, point_feats_dim=64, use_amp=False, amp_dtype=torch.float16):
        super(ProximitySelfAttention, self).__init__()
        self.args = args
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.point_feats_dim = point_feats_dim

        self.get_input_dim()

        # Single input network for self-attention (query, key, value are all the same)
        print("Using single input network for self-attention")
        self.embedding = Embedding(self.input_dim, self.args.embed, additional_dim=self.additional_dim)
        
        encode_layer = nn.TransformerEncoderLayer(d_model=args.embed.output_dim, 
                                                  nhead=args.transformer.num_heads, dim_feedforward=args.transformer.dim_feedforward,
                                                  dropout=args.transformer.dropout, activation=activation_func(args.transformer.act),
                                                  batch_first=True, norm_first=args.transformer.norm_first)
        self.encoder = nn.TransformerEncoder(encode_layer, num_layers=args.transformer.num_layers)
        self.net_blending = MultiheadAttentionRaw(args.embed.output_dim, args.transformer.num_heads, dropout=0, bias=True, batch_first=True)
        
        self.learned_query_token = nn.Parameter(torch.randn(1, 1, args.embed.output_dim), requires_grad=True)
        
        self.score_act = activation_func(args.score_act)

    
    def get_input_dim(self):
        input_dims = {
            1: 3 + 3 + 3,
            2: 3 + 3 + 3,
            3: 3 + 3,
            4: 3 + 1,
            5: 3 + 3 + 3,
            6: 3 + 3 + 3,
            7: 3 + 3 + 3,
            8: 3 + 3 + 3,
            9: 3 + 1,
            10: 3 + 3,
            11: 3 + 3 + 3,
            12: 3,
            13: 3,
            14: 3 + 3 + 3,
            15: 3 + 1 + 1,
        }
        
        # Only need input dimensions for self-attention
        self.input_dim = input_dims[self.args.feat_type]
        self.additional_dim = self.point_feats_dim if self.args.use_point_feats_input else 0
        

    def get_features(self, rays_o, rays_d, points):
        N, H, W, _ = rays_d.shape
        
        rays_d = rays_d.unsqueeze(-2)
        if rays_o.ndim == 2:
            rays_o = rays_o.reshape(N, 1, 1, 1, 3)  # (N, 1, 1, 1, 3)
        else:
            rays_o = rays_o.unsqueeze(-2)

        v = points - rays_o    # (N, 1, 1, num_pts, 3)
        proj = rays_d * torch.sum(v * rays_d, dim=-1).unsqueeze(-1)
        D = v - proj    # (N, H, W, num_pts, 3)

        dists_to_rays = torch.norm(D, dim=-1, keepdim=True)
        proj_dists = torch.norm(proj, dim=-1, keepdim=True)
        
        return proj_dists, dists_to_rays, proj, D
    

    def get_features_input(self, feat_type, rays_o, rays_d, selected_points, vec_pd, vec_d2r, pd, d2r, z):
        N, H, W, K, _ = vec_pd.shape
        if feat_type == 1:
            feature = torch.cat([selected_points.detach(), vec_pd, vec_d2r], dim=-1)
        elif feat_type == 2:
            feature = torch.cat([selected_points.detach(), normalize_vector(vec_pd), vec_d2r], dim=-1)
        elif feat_type == 3:
            feature = torch.cat([selected_points.detach(), vec_d2r], dim=-1)
        elif feat_type == 4:
            feature = torch.cat([selected_points.detach(), d2r], dim=-1)
        elif feat_type == 5:
            feature = torch.cat([selected_points.detach(), vec_pd - max(0, (-z).min()) * rays_d.unsqueeze(-2), vec_d2r], dim=-1)
        elif feat_type == 6:
            feature = torch.cat([selected_points.detach(), vec_pd - pd.min().item() * rays_d.unsqueeze(-2), vec_d2r], dim=-1)
        elif feat_type == 7:
            feature = torch.cat([selected_points.detach(), vec_pd - pd.min(dim=-2, keepdim=True)[0] * rays_d.unsqueeze(-2), vec_d2r], dim=-1)
        elif feat_type == 8:
            feature = torch.cat([selected_points.detach(), vec_pd - (pd.min(dim=-2, keepdim=True)[0] - 1) * rays_d.unsqueeze(-2), vec_d2r], dim=-1)
        elif feat_type == 9:
            vec_o2p = selected_points.detach() - rays_o.reshape(N, 1, 1, 1, 3)
            feature = torch.cat([vec_o2p, torch.sum(vec_o2p * rays_d.unsqueeze(-2), dim=-1, keepdim=True)], dim=-1)
        elif feat_type == 10:
            vec_o2p = selected_points.detach() - rays_o.reshape(N, 1, 1, 1, 3)
            feature = torch.cat([vec_o2p, vec_o2p * rays_d.unsqueeze(-2)], dim=-1)
        elif feat_type == 11:
            feature = torch.cat([selected_points.detach(), vec_pd, vec_d2r + 0.1], dim=-1)
        elif feat_type == 12:
            batch_size = H*W
            rectify_out_dict = rectify_points(selected_points.reshape(N, batch_size, K, 3), 
                                                rays_o.reshape(N, 1, 3).expand(-1, batch_size, -1), 
                                                rays_d.reshape(N, batch_size, 3))
            feature = rectify_out_dict['points_n'].reshape(N, H, W, K, 3)
        elif feat_type == 13:
            batch_size = H*W
            rectify_out_dict = rectify_points(selected_points.reshape(N, batch_size, K, 3), 
                                                rays_o.reshape(N, 1, 3).expand(-1, batch_size, -1), 
                                                rays_d.reshape(N, batch_size, 3),
                                                translate=True,
                                                ts=pd.reshape(N, batch_size, K))
            feature = rectify_out_dict['points_n'].reshape(N, H, W, K, 3)
        elif feat_type == 14:
            batch_size = H*W
            rectify_out_dict = rectify_points(selected_points.reshape(N, batch_size, K, 3), 
                                                rays_o.reshape(N, 1, 3).expand(-1, batch_size, -1), 
                                                rays_d.reshape(N, batch_size, 3),
                                                translate=True,
                                                ts=pd.reshape(N, batch_size, K))
            Rs_w2n = rectify_out_dict['Rs_w2n']
            translation_w2n = rectify_out_dict['translation_w2n']
            points_in_ray_coords = rectify_out_dict['points_n']
            vec_d2r_in_ray_coords = (Rs_w2n.unsqueeze(-3) @ vec_d2r.reshape(N, batch_size, K, 3).unsqueeze(-1)).squeeze(-1)
            vec_pd_in_ray_coords = (Rs_w2n.unsqueeze(-3) @ vec_pd.reshape(N, batch_size, K, 3).unsqueeze(-1)).squeeze(-1)
            feature = torch.cat([points_in_ray_coords.detach(), vec_pd_in_ray_coords, vec_d2r_in_ray_coords], dim=-1).reshape(N, H, W, K, -1)
        elif feat_type == 15:
            batch_size = H*W
            rectify_out_dict = rectify_points(selected_points.reshape(N, batch_size, K, 3), 
                                                rays_o.reshape(N, 1, 3).expand(-1, batch_size, -1), 
                                                rays_d.reshape(N, batch_size, 3),
                                                translate=True,
                                                ts=pd.reshape(N, batch_size, K))
            Rs_w2n = rectify_out_dict['Rs_w2n']
            translation_w2n = rectify_out_dict['translation_w2n']
            points_in_ray_coords = rectify_out_dict['points_n']
            feature = torch.cat([points_in_ray_coords.detach(), pd, d2r], dim=-1).reshape(N, H, W, K, -1)
        else:
            raise ValueError("Unknown k type: {}".format(feat_type))
        return feature


    def forward(self, rays_o, rays_d, points, point_features, points_influ_scores, select_k_ind, selected_points, z=None, c2w=None, bkg_token=None, step=-1, max_step=-1, evaluate=False):

        N, H, W, K, _ = selected_points.shape
        batch_size = N*H*W
        
        pd, d2r, vec_pd, vec_d2r = self.get_features(rays_o, rays_d, selected_points)

        self.pd = pd
        self.d2r = d2r
        self.vec_pd = vec_pd
        self.vec_d2r = vec_d2r
        
        # Get input features
        input_features = self.get_features_input(self.args.feat_type, rays_o, rays_d, selected_points, vec_pd, vec_d2r, pd, d2r, z)
        input_features = input_features.reshape(batch_size, K, -1)
        additional_features = point_features[select_k_ind].reshape(batch_size, K, -1) if self.args.use_point_feats_input else None

        input_features = self.embedding(input_features, additional_features)
        input_features = torch.cat([self.learned_query_token.expand(batch_size, -1, -1), input_features], dim=1)

        out_tokens = self.encoder(input_features)        
        attn_output, _, scores = self.net_blending(query=out_tokens[..., :1, :], 
                                             key=out_tokens[..., 1:, :], 
                                             value=out_tokens[..., 1:, :],
                                             return_raw_logits=True)
        
        value = attn_output if self.args.use_attn_output_as_value else out_tokens[..., 1:, :]
        scores = self.score_act(scores)

        return scores, value, select_k_ind, torch.zeros(0, device=points.device).sum()

