import torch
import torch.nn as nn
from torch import autocast
from torch.nn import functional as F
import tinycudann as tcnn
import math
import random
from .mlp import MLP
from .utils import activation_func, normalize_vector, get_tcnn_init_weights, Encoding, LayerNorm, rectify_points
from .sh import eval_sh, eval_sh_bases, spherical_harmonics
from .transformer import TransformerEncoderLayer, TransformerEncoder
from utils import get_ray_sphere_intersection


C0 = 0.28209479177387814


def safe_print_tensor_stats(name, tensor, step=-1):
    """
    Print tensor statistics without using .item() in the forward pass to avoid graph breaks.
    This function detaches tensors and moves them to CPU before extracting values.
    """
    if step % 201 != 0:
        return
    
    # Detach and move to CPU to avoid graph breaks
    with torch.no_grad():
        tensor_detached = tensor.detach().cpu()
        if tensor_detached.numel() == 1:
            # Single scalar value
            print(f" {name}: {tensor_detached.numpy().item()}")
        else:
            # Multiple values - print shape and statistics
            shape_str = str(tuple(tensor_detached.shape))
            min_val = tensor_detached.min().numpy().item()
            max_val = tensor_detached.max().numpy().item()
            mean_val = tensor_detached.mean().numpy().item()
            if tensor_detached.numel() > 1:
                std_val = tensor_detached.std().numpy().item()
                print(f" {name}: shape={shape_str}, min={min_val:.6f}, max={max_val:.6f}, mean={mean_val:.6f}, std={std_val:.6f}")
            else:
                print(f" {name}: shape={shape_str}, value={min_val:.6f}")


class ProximityAttention(nn.Module):
    def __init__(self, args, point_feats_dim=64, use_amp=False, amp_dtype=torch.float16, attn_act="softmax", attn_act_temp=1.0, coord_scale=1.0):
        super(ProximityAttention, self).__init__()
        self.args = args
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.attn_act = attn_act
        self.attn_act_temp = attn_act_temp
        self.coord_scale = coord_scale
        
        # Unified background sphere config for ray-sphere intersection
        # Use geoms.points.bkg_sphere_radius/center if available, otherwise fallback to defaults
        bkg_sphere_radius = getattr(args, 'bkg_sphere_radius', None)
        if bkg_sphere_radius is None:
            # Try to get from full config path (args might be models.attn subset)
            bkg_sphere_radius = 5.0  # default
        self.bkg_sphere_radius = bkg_sphere_radius * coord_scale
        
        bkg_sphere_center = getattr(args, 'bkg_sphere_center', None)
        if bkg_sphere_center is None:
            bkg_sphere_center = [0.0, 0.0, 0.0]  # default
        self.bkg_sphere_center = [c * coord_scale for c in bkg_sphere_center]

        self.get_kqv_dim()

        if args.shared_network:
            print("Using shared network and key configuration")
            encoder_k = Encoding(self.input_dim_k, args.encode_k_config, args.use_tcnn_encoder)
            encoder_q = Encoding(self.input_dim_q, args.encode_q_config, args.use_tcnn_encoder)
            encoder_v = Encoding(self.input_dim_v, args.encode_v_config, args.use_tcnn_encoder)
            self.encoder_k = encoder_k
            self.encoder_q = encoder_q
            self.encoder_v = encoder_v
            
            assert encoder_k.n_output_dims == encoder_q.n_output_dims == encoder_v.n_output_dims, "Encoder output dims must be the same for shared network"
            assert args.output_dim_k == args.output_dim_q == args.output_dim_v, "Output dims must be the same for shared network"
            
            if args.network_k_config.otype == "MLP":
                shared_network = MLP(encoder_k.n_output_dims, args.network_k_config.n_hidden_layers, \
                                args.network_k_config.n_neurons, args.output_dim_k, \
                                act_type=args.network_k_config.activation, \
                                last_act_type=args.network_k_config.output_activation)
                print("Using MLP for shared network")
            else:
                shared_network = tcnn.Network(encoder_k.n_output_dims, args.output_dim_k, args.network_k_config)
                shared_network.params.data[...] = get_tcnn_init_weights(encoder_k.n_output_dims, args.output_dim_k-1, args.network_k_config, device=shared_network.params.device)
            self.shared_network = shared_network
            
            self.layernorm_pre = LayerNorm(encoder_k.n_output_dims)
            self.layernorm_post = LayerNorm(args.output_dim_k)
            
            self.projection_k = nn.Linear(args.output_dim_k, args.output_dim_k)
            nn.init.xavier_uniform_(self.projection_k.weight)
            
            self.projection_q = nn.Linear(args.output_dim_k, args.output_dim_k)
            nn.init.xavier_uniform_(self.projection_q.weight)
            
        else:
            if args.network_k:
                encoder_k = Encoding(self.input_dim_k, args.encode_k_config, args.use_tcnn_encoder)
                k_list = [encoder_k]
                if args.norm_in_k == "layernorm":  # Somehow very important
                    k_list.append(LayerNorm(encoder_k.n_output_dims))
                if args.network_k_config.otype == "MLP":
                    network_k = MLP(encoder_k.n_output_dims, args.network_k_config.n_hidden_layers, \
                                    args.network_k_config.n_neurons, args.output_dim_k, \
                                    act_type=args.network_k_config.activation, \
                                    last_act_type=args.network_k_config.output_activation)
                    print("Using MLP for network_k")
                else:
                    network_k = tcnn.Network(encoder_k.n_output_dims, args.output_dim_k, args.network_k_config)
                    network_k.params.data[...] = get_tcnn_init_weights(encoder_k.n_output_dims, args.output_dim_k-1, args.network_k_config, device=network_k.params.device)
                k_list.append(network_k)
                if args.norm_out_k == "layernorm":
                    k_list.append(LayerNorm(args.output_dim_k))
                if args.project_k:  # Somehow very important
                    projection = nn.Linear(args.output_dim_k, args.output_dim_k)
                    nn.init.xavier_uniform_(projection.weight)
                    k_list.append(projection)
                    print("Using projection for k")
                self.model_k = nn.Sequential(*k_list)

            if args.use_self_attn:
                self.learnable_token = nn.Parameter(torch.randn(args.output_dim_k), requires_grad=True)
                # Multi-head self-attention over keys (used when self.use_self_attn)
                if self.args.use_self_attn_single_layer:
                    self.self_attn = nn.MultiheadAttention(
                        embed_dim=args.output_dim_k,
                        num_heads=args.transformer.num_heads,
                        batch_first=True,
                        dropout=0.0,
                    )

                else:
                    # Use custom transformer with optional LayerNorm
                    use_norm = getattr(args.transformer, 'use_norm', True)  # Default to True for backward compatibility
                    use_xavier_init = getattr(args.transformer, 'use_xavier_init', False)  # Default to False (use PyTorch default)
                    init_gain = getattr(args.transformer, 'init_gain', 1.0)  # Default to 1.0 for backward compatibility
                    encode_layer = TransformerEncoderLayer(d_model=args.output_dim_k, 
                                                  nhead=args.transformer.num_heads, dim_feedforward=args.transformer.dim_feedforward,
                                                  dropout=args.transformer.dropout, activation=activation_func(args.transformer.act),
                                                  batch_first=True, norm_first=args.transformer.norm_first,
                                                  use_norm=use_norm, use_xavier_init=use_xavier_init, init_gain=init_gain)
                    self.self_attn = TransformerEncoder(encode_layer, num_layers=args.transformer.num_layers)

                if args.network_v_config.otype == "MLP":
                    network_v = MLP(args.output_dim_k, \
                                    args.network_v_config.n_hidden_layers, \
                                    args.network_v_config.n_neurons, args.output_dim_v, \
                                    act_type=args.network_v_config.activation, \
                                    last_act_type=args.network_v_config.output_activation)
                    print("Using MLP for network_v")
                else:
                    network_v = tcnn.Network(args.output_dim_k, \
                                            args.output_dim_v, args.network_v_config)  # Network input dim may differ from encoder output dim
                    network_v.params.data[...] = get_tcnn_init_weights(args.output_dim_k, args.output_dim_v, args.network_v_config, device=network_v.params.device)
                self.model_v = network_v
                
                if args.project_q:
                    projection = nn.Linear(args.output_dim_k, args.output_dim_k)
                    nn.init.xavier_uniform_(projection.weight)
                    self.projection_q = projection
                    print("Using projection for q")
                
                if args.project_k:
                    projection = nn.Linear(args.output_dim_k, args.output_dim_k)
                    nn.init.xavier_uniform_(projection.weight)
                    self.projection_k = projection
                    print("Using projection for k")
                
                if args.predict_hit:
                    activation = activation_func("silu")
                    linear_layer = nn.Linear(args.output_dim_k, 1)
                    nn.init.xavier_uniform_(linear_layer.weight)
                    self.hit_predictor = nn.Sequential(linear_layer, activation)
                    print("Using hit predictor")
                else:
                    self.hit_predictor = None

            else:
                if args.network_q:
                    encoder_q = Encoding(self.input_dim_q, args.encode_q_config, args.use_tcnn_encoder)
                    q_list = [encoder_q]
                    if args.norm_in_q == "layernorm":
                        q_list.append(LayerNorm(encoder_q.n_output_dims))
                    if args.network_q_config.otype == "MLP":
                        network_q = MLP(encoder_q.n_output_dims, args.network_q_config.n_hidden_layers, \
                                        args.network_q_config.n_neurons, args.output_dim_q, \
                                        act_type=args.network_q_config.activation, \
                                        last_act_type=args.network_q_config.output_activation)
                        print("Using MLP for network_q")
                    else:
                        network_q = tcnn.Network(encoder_q.n_output_dims, args.output_dim_q, args.network_q_config)
                        network_q.params.data[...] = get_tcnn_init_weights(encoder_q.n_output_dims, args.output_dim_q, args.network_q_config, device=network_q.params.device)
                    q_list.append(network_q)
                    if args.norm_out_q == "layernorm":
                        q_list.append(LayerNorm(args.output_dim_q))
                    if args.project_q:
                        projection = nn.Linear(args.output_dim_q, args.output_dim_q)
                        nn.init.xavier_uniform_(projection.weight)
                        q_list.append(projection)
                        print("Using projection for q")
                    self.model_q = nn.Sequential(*q_list)

                if args.encode_v_config.otype != "SphericalHarmonics":
                    encoder_v = Encoding(self.input_dim_v, args.encode_v_config, args.use_tcnn_encoder)
                    self.model_v = nn.ModuleList([encoder_v])
                    if args.network_v:
                        if args.network_v_config.otype == "MLP":
                            network_v = MLP(encoder_v.n_output_dims + args.encode_additional_dim_v, \
                                            args.network_v_config.n_hidden_layers, \
                                            args.network_v_config.n_neurons, args.output_dim_v, \
                                            act_type=args.network_v_config.activation, \
                                            last_act_type=args.network_v_config.output_activation)
                            print("Using MLP for network_v")
                        else:
                            network_v = tcnn.Network(encoder_v.n_output_dims + args.encode_additional_dim_v, \
                                                    args.output_dim_v, args.network_v_config)  # Network input dim may differ from encoder output dim
                            network_v.params.data[...] = get_tcnn_init_weights(encoder_v.n_output_dims + args.encode_additional_dim_v, args.output_dim_v, args.network_v_config, device=network_v.params.device)
                        self.model_v.append(network_v)
                    
        if args.prop_network_k:
            prop_encoder_k = Encoding(self.prop_input_dim_k, args.prop_encode_k_config, args.use_tcnn_encoder)
            prop_k_list = [prop_encoder_k]
            if args.norm_in_k == "layernorm":  # Somehow very important
                prop_k_list.append(LayerNorm(prop_encoder_k.n_output_dims))
            if args.prop_network_k_config.otype == "MLP":
                prop_network_k = MLP(prop_encoder_k.n_output_dims, args.prop_network_k_config.n_hidden_layers, \
                                args.prop_network_k_config.n_neurons, args.prop_output_dim_k, \
                                act_type=args.prop_network_k_config.activation, \
                                last_act_type=args.prop_network_k_config.output_activation)
                print("Using Proposal MLP for prop_network_k")
            else:
                prop_network_k = tcnn.Network(prop_encoder_k.n_output_dims, args.prop_output_dim_k, args.prop_network_k_config)
                prop_network_k.params.data[...] = get_tcnn_init_weights(prop_encoder_k.n_output_dims, args.prop_output_dim_k, args.prop_network_k_config, device=prop_network_k.params.device)
            prop_k_list.append(prop_network_k)
            if args.norm_out_k == "layernorm":
                prop_k_list.append(LayerNorm(args.prop_output_dim_k))
            if args.project_k:  # Somehow very important
                projection = nn.Linear(args.prop_output_dim_k, args.prop_output_dim_k)
                nn.init.xavier_uniform_(projection.weight)
                prop_k_list.append(projection)
                print("Using projection for k")
            self.prop_model_k = nn.Sequential(*prop_k_list)

        if args.prop_network_q:
            prop_encoder_q = Encoding(self.prop_input_dim_q, args.prop_encode_q_config, args.use_tcnn_encoder)
            prop_q_list = [prop_encoder_q]
            if args.norm_in_q == "layernorm":  # Somehow very important
                prop_q_list.append(LayerNorm(prop_encoder_q.n_output_dims))
            if args.prop_network_q_config.otype == "MLP":
                prop_network_q = MLP(prop_encoder_q.n_output_dims, args.prop_network_q_config.n_hidden_layers, \
                                args.prop_network_q_config.n_neurons, args.prop_output_dim_q, \
                                act_type=args.prop_network_q_config.activation, \
                                last_act_type=args.prop_network_q_config.output_activation)
                print("Using Proposal MLP for prop_network_q")
            else:
                prop_network_q = tcnn.Network(prop_encoder_q.n_output_dims, args.prop_output_dim_q, args.prop_network_q_config)
                prop_network_q.params.data[...] = get_tcnn_init_weights(prop_encoder_q.n_output_dims, args.prop_output_dim_q, args.prop_network_q_config, device=prop_network_q.params.device)
            prop_q_list.append(prop_network_q)
            if args.norm_out_q == "layernorm":
                prop_q_list.append(LayerNorm(args.prop_output_dim_q))
            if args.project_q:  # Somehow very important
                projection = nn.Linear(args.prop_output_dim_q, args.prop_output_dim_q)
                nn.init.xavier_uniform_(projection.weight)
                prop_q_list.append(projection)
                print("Using projection for k")
            self.prop_model_q = nn.Sequential(*prop_q_list)

        self.score_act = activation_func(args.score_act)

        self.score_scale = nn.Parameter(torch.tensor(args.score_scale, dtype=torch.float32), requires_grad=args.score_scale_learnable)
        
        if self.args.append_bkg_points and self.args.scale_bkg_token_by_num_bkg_points:
            self.bkg_points_scaler = nn.Parameter(torch.tensor(1.0, dtype=torch.float32) * self.args.scale_bkg_token_by_num_bkg_points_init, requires_grad=True)
        else:
            self.bkg_points_scaler = None

    def compute_distance_scale(self, pd, N, H, W):
        """Compute multiplicative distance-based scaling for attention scores.

        pd: (N, H, W, K, 1) raw point-ray projection distances (already positive)
        Returns: (N*H*W, K) scale factors
        Strategies:
          - gamma (default): exp( far_exp + (near_exp - far_exp) * (1 - w^power) )
              Produces slow decay near surface (small w) and accelerated drop for far points (w->1).
              Sharpen controlled via power (>1 => steeper late drop) and near/far exponents.
          - piecewise: different power for near vs far region (threshold) for more aggressive far suppression.
                Hyperparameters (read via getattr with defaults so configs need not define them):
                    dist_gamma (deprecated placeholder) -> use near_exp instead if present.
                    dist_power
                    dist_near_exp, dist_far_exp (exponents at w=0 and w=1 before exp())
                    dist_threshold, dist_power_near, dist_power_far (piecewise)
                Gating:
                    - If per-ray distance spread (max(pd)-min(pd)) <= attn_dist_spread_eps, disable scaling (scale=1).
        """
        eps = 1e-6
        # Normalize over K dimension
        mins = pd.amin(dim=-2, keepdim=True)
        maxs = pd.amax(dim=-2, keepdim=True)
        pd_normalized = (pd - mins) / (maxs - mins + eps)  # 0 for closest, 1 for farthest
        w = pd_normalized.squeeze(-1)  # (N,H,W,K)

        strategy = self.args.get('attn_score_dist_strategy', 'gamma')

        if strategy == 'gamma':
            power = self.args.get('attn_dist_power', 2.0)  # >1 => slower early decay, sharper later
            near_exp = self.args.get('attn_dist_near_exp', 1.0)  # exponent at nearest (scale ~ exp(near_exp))
            far_exp = self.args.get('attn_dist_far_exp', -0.2)   # exponent at farthest (<0 => scale < 1)
            w_pow = w.pow(power)
            inv = 1.0 - w_pow
            exponent = far_exp + (near_exp - far_exp) * inv
            dist_scale = torch.exp(exponent)
        else:
            # Fallback: original behavior with optional sharpening via power
            power = self.args.get('attn_dist_power', 1.0)
            w_pow = w.pow(power)
            dist_scale = torch.exp(1.0 - w_pow)  # matches exp(1) at near, exp(0) at far if power=1

        # Optional gating: if all pd are very close (small spread), disable scaling.
        min_spread = self.args.get('attn_dist_min_spread', None)
        if min_spread is not None and min_spread > 0:
            spread = (maxs - mins)  # (N,H,W,1)
            # print("Distance spread stats before gating:", spread.min().item(), spread.max().item(), spread.mean().item())
            flat_mask = (spread <= min_spread)  # (N,H,W,1)
            if flat_mask.any():
                mask = flat_mask.squeeze(-1).expand_as(w)  # (N,H,W,K)
                dist_scale = torch.where(mask, torch.ones_like(w), dist_scale)

        return dist_scale.reshape(N*H*W, -1)
        
        
    def get_texture_v(self, step, rays_o, rays_d, selected_points, selected_point_features, select_k_ind, texture_map=None, view_independent=False):
        if view_independent:
            assert texture_map is not None, "Texture map is required for view-independent texture embedding"
            N, H, W, _ = texture_map.shape
            v = torch.zeros(N*H*W, self.input_dim_v).to(texture_map.device)
            v = self.model_v[0](v)
            v = torch.cat([v, texture_map.reshape(N*H*W, -1)], dim=-1)
            if self.args.network_v:
                v = self.model_v[1](v).float()
            return v.reshape(N, H, W, -1)

        N, H, W, K, _ = selected_points.shape
        batch_size = N * H * W * K
        vec_pd = self.vec_pd[:N].reshape(batch_size, -1)
        vec_d2r = self.vec_d2r[:N].reshape(batch_size, -1)
        pd = self.pd[:N].reshape(batch_size, -1)
        d2r = self.d2r[:N].reshape(batch_size, -1)
        v = self.get_features_v(self.args.v_type, rays_o, rays_d, selected_points, select_k_ind, vec_pd, vec_d2r, pd, d2r, None)
        v = self.model_v[0](v)
        v = torch.cat([v, selected_point_features.reshape(batch_size, -1)], dim=-1)
        if self.args.network_v:
            v = self.model_v[1](v)
        return v.reshape(N, H, W, K, -1)

    
    def get_kqv_dim(self):
        k_dims = {
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

        q_dims = {
            1: 3,
            2: 3 + 3 + 3,
            3: 3 + 3,
            4: 3 + 3,
            5: 3,
            6: 3 + 3 + 3,
        }

        v_dims = {
            1: 3 + 3,
            2: 3 + 3,
            3: 3,
            4: 0,
            5: 3,
            6: 0,
            7: 3,
            8: 0,
            9: 3,
            10: 3 + 3,
            11: 3 + 3,
            12: 3,
            13: 3,
            14: 3 + 3 + 3,
            15: 3,
            16: 3,
            17: 3 + 3 + 3,
            18: 3 + 3 + 3,
            19: 2,
            20: 3,
            99: 3,   # SH
        }
        
        self.input_dim_k = k_dims[self.args.k_type]
        self.input_dim_q = q_dims[self.args.q_type]
        self.input_dim_v = v_dims[self.args.v_type]

        self.prop_input_dim_k = k_dims[self.args.prop_k_type]
        self.prop_input_dim_q = q_dims[self.args.prop_q_type]


    def get_features(self, rays_o, rays_d, points):
        N, H, W, _ = rays_d.shape
        
        rays_d = rays_d.unsqueeze(-2)
        if rays_o.ndim == 2:
            rays_o = rays_o.reshape(N, 1, 1, 1, 3)  # (N, 1, 1, 1, 3)
        else:
            rays_o = rays_o.unsqueeze(-2)

        # print("rays_o: ", rays_o.shape, "rays_d: ", rays_d.shape, "points: ", points.shape)

        # rays = normalize_vector(rays_d, eps=self.eps).unsqueeze(-2)  # (N, H, W, 1, 3)
        v = points - rays_o    # (N, 1, 1, num_pts, 3)
        proj = rays_d * torch.sum(v * rays_d, dim=-1).unsqueeze(-1)
        D = v - proj    # (N, H, W, num_pts, 3)

        dists_to_rays = torch.norm(D, dim=-1, keepdim=True)
        proj_dists = torch.norm(proj, dim=-1, keepdim=True)

        # print("proj_dists: ", proj_dists.shape, proj_dists.min(), proj_dists.max(), proj_dists.mean())
        # print("dists_to_rays: ", dists_to_rays.shape, dists_to_rays.min(), dists_to_rays.max(), dists_to_rays.mean())
        
        return proj_dists, dists_to_rays, proj, D
    
    def get_bkg_sphere_intersection(self, rays_o, rays_d, sphere_center=None, sphere_radius=None):
        """
        Compute the intersection point of rays with a sphere.
        
        This is a wrapper around the unified get_ray_sphere_intersection function from utils.
        Uses the configured sphere parameters from args by default.
        
        Args:
            rays_o: Ray origins (various shapes supported)
            rays_d: Ray directions (will be normalized)
            sphere_center: (3,) center of sphere, or None to use config (default)
            sphere_radius: scalar radius of the sphere, or None to use config (default)
        
        Returns:
            intersection_points: Intersection points with same spatial shape as rays_d
        """
        if sphere_radius is None:
            sphere_radius = self.bkg_sphere_radius
        if sphere_center is None:
            sphere_center = self.bkg_sphere_center
            
        return get_ray_sphere_intersection(rays_o, rays_d, sphere_center, sphere_radius)

    def get_features_k(self, feat_type, rays_o, rays_d, selected_points, vec_pd, vec_d2r, pd, d2r, z):
        N, H, W, K, _ = vec_pd.shape
        if rays_o.ndim == 2:
            rays_o_hw = rays_o.reshape(N, 1, 1, 3).expand(-1, H, W, -1)
        elif rays_o.ndim == 4:
            rays_o_hw = rays_o
        else:
            raise ValueError(f"Unsupported rays_o ndim={rays_o.ndim}, expected 2 or 4")
        rays_o_expanded = rays_o_hw.unsqueeze(-2)  # (N, H, W, 1, 3)
        rays_o_flat = rays_o_hw.reshape(N, H * W, 3)  # (N, H*W, 3)
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
            vec_o2p = selected_points.detach() - rays_o_expanded
            feature = torch.cat([vec_o2p, torch.sum(vec_o2p * rays_d.unsqueeze(-2), dim=-1, keepdim=True)], dim=-1)
        elif feat_type == 10:
            vec_o2p = selected_points.detach() - rays_o_expanded
            feature = torch.cat([vec_o2p, vec_o2p * rays_d.unsqueeze(-2)], dim=-1)
        elif feat_type == 11:
            feature = torch.cat([selected_points.detach(), vec_pd, vec_d2r + 0.1], dim=-1)
        elif feat_type == 12:
            batch_size = H*W
            rectify_out_dict = rectify_points(selected_points.reshape(N, batch_size, K, 3), 
                                                rays_o_flat, 
                                                rays_d.reshape(N, batch_size, 3))
            feature = rectify_out_dict['points_n'].reshape(N, H, W, K, 3)
        elif feat_type == 13:
            batch_size = H*W
            rectify_out_dict = rectify_points(selected_points.reshape(N, batch_size, K, 3), 
                                                rays_o_flat, 
                                                rays_d.reshape(N, batch_size, 3),
                                                translate=True,
                                                ts=pd.reshape(N, batch_size, K))
            feature = rectify_out_dict['points_n'].reshape(N, H, W, K, 3)
        elif feat_type == 14:
            batch_size = H*W
            rectify_out_dict = rectify_points(selected_points.reshape(N, batch_size, K, 3), 
                                                rays_o_flat, 
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
                                                rays_o_flat, 
                                                rays_d.reshape(N, batch_size, 3),
                                                translate=True,
                                                ts=pd.reshape(N, batch_size, K))
            Rs_w2n = rectify_out_dict['Rs_w2n']
            translation_w2n = rectify_out_dict['translation_w2n']
            points_in_ray_coords = rectify_out_dict['points_n'].reshape(N, H, W, K, -1)
            feature = torch.cat([points_in_ray_coords.detach(), pd, d2r], dim=-1).reshape(N, H, W, K, -1)
        else:
            raise ValueError("Unknown k type: {}".format(feat_type))
        return feature
    

    def get_features_q(self, feat_type, rays_o, rays_d, selected_points, vec_pd, vec_d2r, pd, d2r, z):
        N, H, W, _ = rays_d.shape
        if rays_o.ndim == 2:
            rays_o_hw = rays_o.reshape(N, 1, 1, 3).expand(-1, H, W, -1)
        elif rays_o.ndim == 4:
            rays_o_hw = rays_o
        else:
            raise ValueError(f"Unsupported rays_o ndim={rays_o.ndim}, expected 2 or 4")
        if feat_type == 1:
            feature = rays_d
        elif feat_type == 2:
            feature = torch.cat([selected_points.detach(), vec_pd, vec_d2r], dim=-1)            
        elif feat_type == 3:
            # Plucker coordinates
            feature = torch.cat([torch.cross(rays_o_hw, rays_d, dim=-1), rays_d], dim=-1)  # (N, H, W, 6)
        elif feat_type == 4:
            feature = torch.cat([rays_o_hw, rays_d], dim=-1)  # (N, H, W, 6)
        elif feat_type == 5:
            feature = torch.ones_like(rays_d)
        elif feat_type == 6:
            # Self attention
            feature = torch.cat([selected_points.detach(), vec_pd, vec_d2r], dim=-1)
        else:
            raise ValueError("Unknown q type: {}".format(feat_type))
        return feature
    

    def get_features_v(self, feat_type, rays_o, rays_d, points, select_k_ind, vec_pd, vec_d2r, pd, d2r, z):
        N, H, W, K = select_k_ind.shape[:4]
        if rays_o.ndim == 2:
            rays_o_hw = rays_o.reshape(N, 1, 1, 3).expand(-1, H, W, -1)
            rays_o_points = rays_o
        elif rays_o.ndim == 4:
            rays_o_hw = rays_o
            # Point-wise features still need one camera origin per image.
            rays_o_points = rays_o[:, 0, 0, :]
        else:
            raise ValueError(f"Unsupported rays_o ndim={rays_o.ndim}, expected 2 or 4")
        rays_o_expanded = rays_o_hw.unsqueeze(-2)  # (N, H, W, 1, 3)
        rays_o_flat = rays_o_hw.reshape(N, H * W, 3)  # (N, H*W, 3)
        if feat_type == 1:
            feature = torch.cat([vec_pd, vec_d2r], dim=-1)
        elif feat_type == 2:
            feature = torch.cat([normalize_vector(vec_pd), vec_d2r], dim=-1)
        elif feat_type == 3:
            feature = vec_d2r
        elif feat_type in [4, 6, 8]:
            feature = torch.empty(0, device=points.device)
        elif feat_type in [5, 7, 9]:
            feature = points
        elif feat_type == 10:
            points = points[select_k_ind].detach()
            rays_to_points = points - rays_o_expanded   # (N, H, W, K, 3)
            rays_to_points = rays_to_points / torch.norm(rays_to_points, dim=-1, keepdim=True)
            feature = torch.cat([rays_to_points, points], dim=-1).reshape(-1, 6)
        elif feat_type == 11:
            points = points[select_k_ind].detach()
            rays_to_points = points - rays_o_expanded   # (N, H, W, K, 3)
            feature = torch.cat([rays_to_points, points], dim=-1).reshape(-1, 6)
        elif feat_type == 12:
            points = points[select_k_ind].detach()
            feature = points - rays_o_expanded   # (N, H, W, K, 3)
            feature = feature.reshape(-1, 3)
        elif feat_type == 13:
            points = points[select_k_ind].detach()
            rays_to_points = points - rays_o_expanded   # (N, H, W, K, 3)
            feature = rays_to_points / torch.norm(rays_to_points, dim=-1, keepdim=True)
            feature = feature.reshape(-1, 3)
        elif feat_type == 14:
            feature = torch.cat([points[select_k_ind].detach().reshape(-1, 3), vec_pd, vec_d2r], dim=-1)
        elif feat_type == 15:
            batch_size = H*W
            selected_points = points[select_k_ind].detach()
            rectify_out_dict = rectify_points(selected_points.reshape(N, batch_size, K, 3), 
                                                rays_o_flat, 
                                                rays_d.reshape(N, batch_size, 3))
            feature = rectify_out_dict['points_n'].reshape(N*H*W*K, 3)
        elif feat_type == 16:
            batch_size = H*W
            selected_points = points[select_k_ind].detach()
            rectify_out_dict = rectify_points(selected_points.reshape(N, batch_size, K, 3), 
                                                rays_o_flat, 
                                                rays_d.reshape(N, batch_size, 3),
                                                translate=True,
                                                ts=pd.reshape(N, batch_size, K))
            feature = rectify_out_dict['points_n'].reshape(N*H*W*K, 3)
        elif feat_type == 17:
            batch_size = H*W
            selected_points = points[select_k_ind].detach()
            rectify_out_dict = rectify_points(selected_points.reshape(N, batch_size, K, 3), 
                                                rays_o_flat, 
                                                rays_d.reshape(N, batch_size, 3),
                                                translate=True,
                                                ts=pd.reshape(N, batch_size, K))
            Rs_w2n = rectify_out_dict['Rs_w2n']
            translation_w2n = rectify_out_dict['translation_w2n']
            points_in_ray_coords = rectify_out_dict['points_n']
            vec_d2r_in_ray_coords = (Rs_w2n.unsqueeze(-3) @ vec_d2r.reshape(N, batch_size, K, 3).unsqueeze(-1)).squeeze(-1)
            vec_pd_in_ray_coords = (Rs_w2n.unsqueeze(-3) @ vec_pd.reshape(N, batch_size, K, 3).unsqueeze(-1)).squeeze(-1)
            feature = torch.cat([points_in_ray_coords.detach(), vec_pd_in_ray_coords, vec_d2r_in_ray_coords], dim=-1).reshape(N*H*W*K, -1)
        elif feat_type == 18:
            batch_size = H*W
            selected_points = points[select_k_ind].detach()
            rectify_out_dict = rectify_points(selected_points.reshape(N, batch_size, K, 3), 
                                                rays_o_flat, 
                                                rays_d.reshape(N, batch_size, 3),
                                                translate=True,
                                                ts=pd.reshape(N, batch_size, K))
            Rs_w2n = rectify_out_dict['Rs_w2n']
            translation_w2n = rectify_out_dict['translation_w2n']
            points_in_ray_coords = rectify_out_dict['points_n'].reshape(-1, 3)
            feature = torch.cat([points_in_ray_coords.detach(), vec_pd, vec_d2r], dim=-1)
        elif feat_type == 19:
            batch_size = H*W
            selected_points = points[select_k_ind].detach()
            rectify_out_dict = rectify_points(selected_points.reshape(N, batch_size, K, 3), 
                                                rays_o_flat, 
                                                rays_d.reshape(N, batch_size, 3))
            feature = rectify_out_dict['points_n'].reshape(N*H*W*K, 3)[:, :2]
        elif feat_type == 20:
            points = points[select_k_ind].detach()
            feature = points - rays_o_expanded   # (N, H, W, K, 3)
            feature = normalize_vector(feature).reshape(-1, 3)
        else:
            raise ValueError("Unknown v type: {}".format(feat_type))
        return feature
    

    def top_k_gumbel(self, logits, k, temperature=0.5, dim=-1):
        top_k_logits = torch.zeros_like(logits)

        for _ in range(k):
            top1_gumbel = F.gumbel_softmax(logits, tau=temperature, hard=True, dim=dim)
            top_k_logits += top1_gumbel
            logits = logits - top1_gumbel * 1e10

        return top_k_logits, torch.topk(top_k_logits, k, dim=dim, sorted=False)[1]


    # @torch.autocast(device_type="cuda")
    def forward(self, rays_o, rays_d, points, point_features, points_influ_scores, select_k_ind, z=None, c2w=None, bkg_token=None, step=-1, max_step=-1, evaluate=False, scores_only=False, append_bkg_points_feats=None, points_scaler=None, bkg_points_mask=None, bkg_depth_offset=None):

        selected_points = points[select_k_ind]
        if self.args.append_bkg_points:
            # Compute ray-sphere intersection point using unified config
            sphere_intersection = self.get_bkg_sphere_intersection(rays_o, rays_d)
            
            # Apply depth offset from cubemap if provided
            # The depth offset moves the intersection point along the ray direction
            if bkg_depth_offset is not None:
                # bkg_depth_offset: (N, H, W, 1) - offset distance along ray
                # Normalize ray direction for offset application
                rays_d_normalized = F.normalize(rays_d, dim=-1)
                # Move the intersection point along the ray by the offset amount
                sphere_intersection = sphere_intersection + rays_d_normalized * bkg_depth_offset
            
            # Append sphere intersection point to selected_points
            # selected_points: (N, H, W, K, 3), sphere_intersection: (N, H, W, 3)
            sphere_intersection = sphere_intersection.unsqueeze(-2)  # (N, H, W, 1, 3)
            self.sphere_intersection = sphere_intersection
            selected_points = torch.cat([selected_points, sphere_intersection], dim=-2)  # (N, H, W, K+1, 3)
            assert append_bkg_points_feats is not None, "append_bkg_points_feats is required when append_bkg_points is True"
        
        N, H, W, K, _ = selected_points.shape
        # with autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
        pd, d2r, vec_pd, vec_d2r = self.get_features(rays_o, rays_d, selected_points)

        # Apply point scaler if provided
        if points_scaler is not None:
            # points_scaler: (num_points, 1), select_k_ind: (N, H, W, K)
            selected_scaler = points_scaler[select_k_ind]  # (N, H, W, K, 1)
            if self.args.append_bkg_points:
                # For appended background point, use scaler of 1.0 (no scaling)
                bkg_scaler = torch.ones(N, H, W, 1, 1, device=selected_scaler.device)
                selected_scaler = torch.cat([selected_scaler, bkg_scaler], dim=-2)  # (N, H, W, K+1, 1)
            # Scale the features
            # pd: (N, H, W, K, 1), d2r: (N, H, W, K, 1)
            # vec_pd: (N, H, W, K, 3), vec_d2r: (N, H, W, K, 3)
            d2r = d2r * selected_scaler
            vec_d2r = vec_d2r * selected_scaler

            if not self.args.scale_d2r_only:
                vec_pd = vec_pd * selected_scaler
                pd = pd * selected_scaler

        self.pd = pd
        self.d2r = d2r
        self.vec_pd = vec_pd
        self.vec_d2r = vec_d2r
        
        if self.args.add_random_shift_to_vec_pd:
            noise = torch.rand(N, H, W, 1, 1, device=vec_pd.device)
            min_pd = pd.min(dim=-2, keepdim=True)[0]
            vec_pd = vec_pd - min_pd * rays_d.unsqueeze(-2)
        
        hit_pred = None
        
        if self.args.shared_network:
            feature_k = self.get_features_k(self.args.k_type, rays_o, rays_d, selected_points, vec_pd, vec_d2r, pd, d2r, z)
            feature_q = self.get_features_q(self.args.q_type, rays_o, rays_d, selected_points, vec_pd, vec_d2r, pd, d2r, z)
            feature_v = self.get_features_v(self.args.v_type, rays_o, rays_d, selected_points, vec_pd, vec_d2r, pd, d2r, z)
            
            encoded_k = self.encoder_k(feature_k.flatten(0, -2)) # (H*W*K, D_k)
            encoded_q = self.encoder_q(feature_q.flatten(0, -2)) # (H*W, D_q)
            encoded_v = self.encoder_v(feature_v.flatten(0, -2)) # (H*W*K, D_v)
            
            # # Randomly zero v during training
            # if hasattr(self.args, 'v_zero_prob') and self.args.v_zero_prob > 0 and not evaluate:
            #     if torch.rand(1).item() < self.args.v_zero_prob:
            #         encoded_v = torch.zeros_like(encoded_v)
            #         if step % 201 == 0:
            #             print(f"Randomly zeroed v with probability {self.args.v_zero_prob}")
            
            num_k = encoded_k.shape[0]
            num_q = encoded_q.shape[0]
            num_v = encoded_v.shape[0]
            
            encoded_kq = torch.cat([encoded_k, encoded_q], dim=0)
            normed_kq = self.layernorm_pre(encoded_kq)
            
            encoded_input = torch.cat([normed_kq, encoded_v], dim=0)
            output = self.shared_network(encoded_input)
            
            normed_output_kq = self.layernorm_post(output[:num_k + num_q])
            projected_k = self.projection_k(normed_output_kq[:num_k])
            projected_q = self.projection_q(normed_output_kq[num_k:])
            
            v = output[num_k + num_q:]
            
            _, D_kq = projected_q.shape
            if bkg_token is not None and not self.args.append_bkg_points:
                projected_k = projected_k.reshape(N*H*W, -1, D_kq)
                projected_k = torch.cat([projected_k, bkg_token.unsqueeze(0).expand(N*H*W, -1).unsqueeze(1)], dim=1)
                
            scores = torch.einsum('ikj,ikj->ik', projected_q.reshape(N*H*W, -1, D_kq), projected_k.reshape(N*H*W, -1, D_kq)) / math.sqrt(D_kq)
            self.raw_scores = scores

            scaled_scores = scores * self.score_scale * self.args.score_scale_factor

            # exit(0)
            if self.args.get('scale_scores_by_dist', False):
                dist_scale = self.compute_distance_scale(pd, N, H, W)
                if bkg_token is not None and not self.args.append_bkg_points:
                    dist_scale = torch.cat([dist_scale, torch.ones(N*H*W, 1, device=dist_scale.device)], dim=-1)
                scaled_scores = scaled_scores * dist_scale
                if step % 201 == 0:
                    safe_print_tensor_stats("dist_scale(shared)", dist_scale, step)

            scores = self.score_act(scaled_scores)
            self.act_scores = scores
            
            if scores_only:
                return scores
            
            reg_loss = torch.zeros(0, device=points.device).sum()
        
        else:
            if self.args.prop_network_k:
                prop_k = self.get_features_k(self.args.prop_k_type, rays_o, rays_d, selected_points.detach(), vec_pd, vec_d2r, pd, d2r, z)
                prop_k = self.prop_model_k(prop_k.flatten(0, -2)) # (H*W*K, D_k)

                if self.args.prop_network_q:
                    prop_q = self.get_features_q(self.args.prop_q_type, rays_o, rays_d, selected_points.detach(), vec_pd, vec_d2r, pd, d2r, z)
                    prop_q = self.prop_model_q(prop_q.flatten(0, -2))

                    _, D_kq = prop_q.shape
                    prop_scores = torch.einsum('ij,ikj->ik', prop_q, prop_k.reshape(N*H*W, -1, D_kq)) / math.sqrt(D_kq)
                else:
                    D_kq = prop_k.shape[-1]
                    K = select_k_ind.shape[-1]
                    prop_scores = prop_k.reshape(-1, K, D_kq).mean(dim=-1)

                select_k = prop_scores.shape[-1]
                # if self.args.select_prop_scores and self.args.select_prop_scores_topk < select_k:
                N, H, W, _ = select_k_ind.shape
                # prop_scores_orig = prop_scores.clone()
                prop_scores = prop_scores * points_influ_scores[select_k_ind].detach().reshape(-1, select_k)
                # prop_scores = prop_scores * points_influ_scores[select_k_ind].reshape(-1, select_k)
                prop_scores_orig = prop_scores.clone()
                # scores, topk_ind = torch.topk(prop_scores, self.args.select_prop_scores_topk, dim=-1, sorted=False)
                # _, topk_ind = self.top_k_gumbel(prop_scores, self.args.select_prop_scores_topk, temperature=0.5, dim=-1)
                prop_probabilites = F.softmax(prop_scores, dim=-1)
                prop_probabilites = torch.pow(prop_probabilites, (self.args.select_prop_b * step / max_step) / ((self.args.select_prop_b - 1) * step / max_step + 1))
                topk_ind = torch.multinomial(prop_probabilites, self.args.select_prop_scores_topk, replacement=False)
                # scores = torch.gather(prop_scores * topk_logits, -1, topk_ind)
                select_k_ind = torch.gather(select_k_ind.reshape(-1, select_k), -1, topk_ind).reshape(N, H, W, -1)
                vec_pd = torch.gather(vec_pd.flatten(0, 2), 1, topk_ind.unsqueeze(-1).expand(-1, -1, 3)).reshape(N, H, W, -1, 3)
                vec_d2r = torch.gather(vec_d2r.flatten(0, 2), 1, topk_ind.unsqueeze(-1).expand(-1, -1, 3)).reshape(N, H, W, -1, 3)
                # pd = torch.gather(pd.flatten(0, 2), 1, topk_ind.unsqueeze(-1).expand(-1, -1, 1)).reshape(N, H, W, -1, 1)
                # d2r = torch.gather(d2r.flatten(0, 2), 1, topk_ind.unsqueeze(-1).expand(-1, -1, 1)).reshape(N, H, W, -1, 1)

            k = self.get_features_k(self.args.k_type, rays_o, rays_d, selected_points, vec_pd, vec_d2r, pd, d2r, z)        
            klen = k.shape[-2]
            k = self.model_k(k.flatten(0, -2)) # (N*H*W*K, D_k)

            if self.args.use_self_attn:
                # Multi-head self-attention on keys only; no query used
                D_kq = k.shape[-1]
                Kp = select_k_ind.shape[-1]
                k_seq = k.reshape(N*H*W, Kp, D_kq)
                k_seq = torch.cat([self.learnable_token.unsqueeze(0).expand(N*H*W, -1).unsqueeze(1), k_seq], dim=1)
                Kp = Kp + 1
                if self.args.use_self_attn_single_layer:
                    # Self-attn: Q=K=V=k_seq; attn weights shape (B, Kp, Kp)
                    attn_out = self.self_attn(k_seq, k_seq, k_seq, need_weights=False)[0]
                else:
                    attn_out = self.self_attn(k_seq)
                query = attn_out[..., 0, :]
                key = attn_out[..., 1:, :]
                
                if self.hit_predictor is not None:
                    hit_pred = self.hit_predictor(query)
                    hit_pred = torch.sigmoid(hit_pred)
                    hit_pred = hit_pred.reshape(N, H, W)
                
                value = self.model_v(key.reshape(-1, D_kq))
                
                if bkg_token is not None and not self.args.append_bkg_points:
                    key = torch.cat([key, bkg_token.unsqueeze(0).expand(N*H*W, -1).unsqueeze(1)], dim=1)
                    
                if self.args.project_k:
                    key = self.projection_k(key)
                    
                if self.args.project_q:
                    query = self.projection_q(query)
                    
                scores = torch.einsum('ij,ikj->ik', query, key.reshape(N*H*W, -1, D_kq)) / math.sqrt(D_kq)
                self.raw_scores = scores
                safe_print_tensor_stats("self-attn mha: scores bf act", scores, step)
                scores = self.score_act(scores * self.args.score_scale_factor)
                self.act_scores = scores
                safe_print_tensor_stats("self-attn mha: scores af act", scores, step)

                if scores_only:
                    return scores

                return scores, value, select_k_ind, torch.zeros(0, device=points.device).sum(), hit_pred

            if self.args.network_q:
                q = self.get_features_q(self.args.q_type, rays_o, rays_d, selected_points, vec_pd, vec_d2r, pd, d2r, z)
                q = self.model_q(q.flatten(0, -2)) # (N*H*W, D_q)

                _, D_kq = q.shape
                
                if bkg_token is not None and not self.args.append_bkg_points:
                    k = k.reshape(N*H*W, -1, D_kq)
                    k = torch.cat([k, bkg_token.unsqueeze(0).expand(N*H*W, -1).unsqueeze(1)], dim=1)

                if self.attn_act == "softpick":
                    qlen = 1
                    default_scale = 1.0 / (D_kq ** 0.5)
                    mask = torch.tril(torch.ones(klen, klen, device=k.device))
                    scores = torch.einsum('ij,ikj->ik', q, k.reshape(N*H*W, -1, D_kq))
                    scores = scores * self.attn_act_temp * default_scale
                    scores = scores.masked_fill(mask[klen-qlen:klen, :klen] == 0, float('-inf'))
                    # print(mask)
                    # print(mask[klen-qlen:klen, :klen] == 0)
                    # exit(0)
                    # print("scores: ", scores.shape, scores.min().item(), scores.max().item(), scores.mean().item())
                else:
                    scores = torch.einsum('ij,ikj->ik', q, k.reshape(N*H*W, -1, D_kq)) / math.sqrt(D_kq)

                safe_print_tensor_stats("q vector", q, step)
                safe_print_tensor_stats("scores", scores, step)
            else:
                D_kq = k.shape[-1]
                K = select_k_ind.shape[-1]
                scores = k.reshape(-1, K, D_kq).mean(dim=-1)
            
            if self.args.use_score_scaler:
                score_scaler = k[..., -1]
                k = k[..., :-1]
                if step % 201 == 0:
                    print(" score_scaler: ", score_scaler.shape, score_scaler.min(), score_scaler.max(), score_scaler.mean())
                scores = scores * score_scaler.reshape(N*H*W, -1)

            reg_loss = torch.zeros(0, device=points.device).sum()
            if self.args.prop_network_k:
                padded_scores = torch.zeros_like(prop_scores_orig)
                padded_scores.scatter_(1, topk_ind, scores * points_influ_scores[select_k_ind].reshape(-1, select_k_ind.shape[-1]))
                padded_scores = padded_scores.detach()
                if (self.args.prop_loss_type).lower() == "mse":
                    reg_loss = F.mse_loss(F.softmax(prop_scores_orig, dim=-1), F.softmax(padded_scores, dim=-1))
                elif (self.args.prop_loss_type).lower() == "kl":
                    reg_loss = F.kl_div(F.log_softmax(prop_scores_orig, dim=-1), F.softmax(padded_scores, dim=-1), reduction='batchmean')
                elif (self.args.prop_loss_type).lower() == "chisquare":
                    prop_scores_prob = F.softmax(prop_scores_orig, dim=-1)
                    padded_scores_prob = F.softmax(padded_scores, dim=-1)
                    reg_loss = torch.mean((padded_scores_prob - prop_scores_prob) ** 2 / (padded_scores_prob + prop_scores_prob + 1e-6))

            self.raw_scores = scores

            scaled_scores = scores * self.score_scale * self.args.score_scale_factor

            if self.args.get('scale_scores_by_dist', False):
                dist_scale = self.compute_distance_scale(pd, N, H, W)
                if bkg_token is not None and not self.args.append_bkg_points:
                    dist_scale = torch.cat([dist_scale, torch.ones(N*H*W, 1, device=dist_scale.device)], dim=1)
                scaled_scores = scaled_scores * dist_scale
                if step % 201 == 0:
                    safe_print_tensor_stats("dist_scale", dist_scale, step)
                    
            if self.args.append_bkg_points and self.args.scale_bkg_token_by_num_bkg_points:
                selected_bkg_points_mask = bkg_points_mask[select_k_ind]
                num_fg_points_normalized = selected_bkg_points_mask.sum(dim=-2) * self.bkg_points_scaler / K
                scaled_scores[..., -1] = scaled_scores[..., -1] - num_fg_points_normalized.reshape(N*H*W)
                if step % 201 == 0:
                    print(" bkg_points_scaler: ", self.bkg_points_scaler.item())

            scores = self.score_act(scaled_scores)
            self.act_scores = scores

            if step % 201 == 0:
                print(" score_scale: ", self.score_scale.item())

            if scores_only:
                return scores

            if self.args.use_pc_feats_directly:
                if self.args.use_sh:
                    if rays_o.ndim == 2:
                        rays_o_expanded = rays_o.reshape(N, 1, 1, 1, 3)
                    else:
                        rays_o_expanded = rays_o.unsqueeze(-2)
                    rays_to_points = selected_points - rays_o_expanded
                    num_bases = (self.args.sh_degree + 1) ** 2
                    D = point_features.shape[-1]
                    selected_point_features = point_features[select_k_ind].reshape(N, H, W, -1, num_bases, D)
                    if self.args.append_bkg_points:
                        # Handle both shared features (1, D) and per-ray cubemap features (N, H, W, 1, D)
                        if append_bkg_points_feats.dim() == 5:
                            # Per-ray features from cubemap: (N, H, W, 1, D)
                            bkg_feats_expanded = append_bkg_points_feats.reshape(N, H, W, 1, num_bases, D)
                        else:
                            # Shared features: (1, D) or (num_feats, D)
                            bkg_feats_expanded = append_bkg_points_feats.squeeze(0).expand(N, H, W, 1, -1).reshape(N, H, W, 1, num_bases, D)
                        selected_point_features = torch.cat([selected_point_features, bkg_feats_expanded], dim=-3)
                    # Clamp the step so a sentinel/negative step can never ask for a
                    # negative SH degree, which allocates a zero-width basis tensor.
                    sh_degree_to_use = int(min(max(step, 0) // self.args.sh_degree_interval, self.args.sh_degree))
                    v = spherical_harmonics(sh_degree_to_use, rays_to_points, selected_point_features)
                    if step % 201 == 0:
                        print("v", v.shape, v.min(), v.max(), selected_point_features.shape, selected_point_features.min(), selected_point_features.max())
                    v = activation_func(self.args.pc_feats_act)(v)
                else:
                    v = activation_func(self.args.pc_feats_act)(point_features)[select_k_ind].reshape(N, H, W, -1, point_features.shape[-1])
                    if self.args.append_bkg_points:
                        # Handle both shared features (1, D) and per-ray cubemap features (N, H, W, 1, D)
                        if append_bkg_points_feats.dim() == 5:
                            # Per-ray features from cubemap: (N, H, W, 1, D)
                            bkg_feats_expanded = append_bkg_points_feats
                        else:
                            # Shared features: (1, D) or (num_feats, D)
                            bkg_feats_expanded = append_bkg_points_feats.squeeze(0).expand(N, H, W, 1, -1)
                        v = torch.cat([v, bkg_feats_expanded], dim=-2)
            else:
                if self.args.encode_v_config.otype == "SphericalHarmonics":
                    rays_o_points = rays_o[:, 0, 0, :] if rays_o.ndim == 4 else rays_o
                    rays_to_points = points - rays_o_points   # (N, 3)
                    rays_to_points = rays_to_points / (torch.norm(rays_to_points, dim=-1, keepdim=True) + 1e-6)
                    N = rays_to_points.shape[0]
                    # sh_bases = self.model_v[0]((rays_to_points + 1) / 2)    # See https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md#spherical-harmonics
                    # # point_features should be SH coefficients (N, 3*D_sh), sh_bases should be (N, D_sh)
                    # N, D_sh = sh_bases.shape
                    # # sh = point_features.reshape(N, 3, -1) * sh_bases.unsqueeze(1)   # (N, 3, D_sh) * (N, 1, D_sh) = (N, 3, D_sh)
                    sh = eval_sh(self.args.encode_v_config.degree-1, point_features.reshape(N, 3, -1), rays_to_points)
                    if step % 201 == 0:
                        print(" sh bf act: ", sh.shape, sh.min().item(), sh.max().item(), sh.mean().item())
                    # sh = torch.sigmoid(sh / 10 + 0.5)
                    sh = sh / 10 + 0.5
                    if step % 201 == 0:
                        print(" sh af act: ", sh.shape, sh.min().item(), sh.max().item(), sh.mean().item())

                    if self.args.network_v:
                        v = self.model_v[1](sh)
                    else:
                        v = sh
                        # v = (sh.sum(dim=-1) * C0 + 0.5).clamp(min=0)
                        # v = torch.sigmoid(sh.sum(dim=-1) + 0.5)
                    # if step % 200 == 0:
                    #     print("v: ", v.shape, v.min(), v.max(), v.mean())
                    v = v[select_k_ind.reshape(-1)]

                else:
                    select_k = scores.shape[-1]
                    if self.args.select_v_scores:
                        if self.args.select_v_scores_rnd and not evaluate:
                            select_v_scores_topk = random.randint(self.args.select_v_scores_topk, self.args.select_v_scores_topk_max)
                        elif self.args.select_v_scores_rnd and evaluate:
                            select_v_scores_topk = self.args.select_v_scores_topk_max
                        else:
                            select_v_scores_topk = self.args.select_v_scores_topk
                        if select_v_scores_topk < select_k:
                            N, H, W, _ = select_k_ind.shape
                            # if self.args.select_v_scores_softmax:
                            #     _, topk_ind = torch.topk(scores.softmax(dim=-1), select_v_scores_topk, dim=-1, sorted=False)
                            #     scores = torch.gather(scores, -1, topk_ind)
                            # else:
                            scores, topk_ind = torch.topk(scores, select_v_scores_topk, dim=-1, sorted=False)
                            select_k_ind = torch.gather(select_k_ind.reshape(-1, select_k), -1, topk_ind).reshape(N, H, W, -1)
                            vec_pd = torch.gather(vec_pd.flatten(0, 2), 1, topk_ind.unsqueeze(-1).expand(-1, -1, 3))
                            vec_d2r = torch.gather(vec_d2r.flatten(0, 2), 1, topk_ind.unsqueeze(-1).expand(-1, -1, 3))

                    vec_pd = vec_pd.reshape(-1, 3)
                    vec_d2r = vec_d2r.reshape(-1, 3)

                    v = self.get_features_v(self.args.v_type, rays_o, rays_d, points, select_k_ind, vec_pd, vec_d2r, pd, d2r, z)
                    
                    # # Randomly zero v during training
                    # if hasattr(self.args, 'v_zero_prob') and self.args.v_zero_prob > 0 and not evaluate:
                    #     if torch.rand(1).item() < self.args.v_zero_prob:
                    #         v = torch.zeros_like(v)
                    #         if step % 201 == 0:
                    #             print(f"Randomly zeroed v with probability {self.args.v_zero_prob}")
                    
                    selected_point_features = point_features[select_k_ind]
                    if self.args.append_bkg_points:
                        # Handle both shared features (1, D) and per-ray cubemap features (N, H, W, 1, D)
                        if append_bkg_points_feats.dim() == 5:
                            # Per-ray features from cubemap: (N, H, W, 1, D)
                            bkg_feats_expanded = append_bkg_points_feats
                        else:
                            # Shared features: (1, D) or (num_feats, D)
                            bkg_feats_expanded = append_bkg_points_feats.squeeze(0).expand(N, H, W, 1, -1)
                        selected_point_features = torch.cat([selected_point_features, bkg_feats_expanded], dim=-2)
                    selected_point_features = selected_point_features.reshape(-1, selected_point_features.shape[-1])
                    
                    if self.args.v_type in [4, 5, 10, 11, 12, 13]:
                        if v.nelement() > 0:
                            v = self.model_v[0](v)
                        v = torch.cat([v, selected_point_features], dim=-1)
                        if self.args.network_v:
                            v = self.model_v[1](v)
                    elif self.args.v_type in [6, 7]:
                        if v.nelement() > 0:
                            v = self.model_v[0](v)
                        rays_o_points = rays_o[:, 0, 0, :] if rays_o.ndim == 4 else rays_o
                        rays_to_points = points - rays_o_points   # (N, 3)
                        rays_to_points = rays_to_points / (torch.norm(rays_to_points, dim=-1, keepdim=True) + 1e-6)
                        N = rays_to_points.shape[0]
                        C = point_features.shape[-1] // 3
                        D = int(math.sqrt(C))
                        sh = eval_sh(D-1, point_features.reshape(N, 3, -1), rays_to_points)
                        v = torch.cat([v, sh.reshape(N, -1)], dim=-1)
                        if self.args.network_v:
                            v = self.model_v[1](v)[select_k_ind.reshape(-1)]
                    elif self.args.v_type in [8, 9]:
                        if v.nelement() > 0:
                            v = self.model_v[0](v)
                        rays_o_points = rays_o[:, 0, 0, :] if rays_o.ndim == 4 else rays_o
                        rays_to_points = points - rays_o_points   # (N, 3)
                        rays_to_points = rays_to_points / (torch.norm(rays_to_points, dim=-1, keepdim=True) + 1e-6)
                        N = rays_to_points.shape[0]
                        C = point_features.shape[-1] // 3
                        sh_bases = eval_sh_bases(C, rays_to_points)
                        sh = (point_features.reshape(N, 3, -1) * sh_bases.unsqueeze(1)).reshape(N, -1)
                        v = torch.cat([v, sh], dim=-1)
                        if self.args.network_v:
                            v = self.model_v[1](v)[select_k_ind.reshape(-1)]
                    else:
                        v = self.model_v[0](v) # (H*W*K, D_v)
                        v = torch.cat([v, selected_point_features], dim=-1)
                        if self.args.network_v:
                            v = self.model_v[1](v)
                    
        return scores, v, select_k_ind, reg_loss, hit_pred
