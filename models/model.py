import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast
import os
import time
import numpy as np
import random
import open3d as o3d
from pytorch3d.ops import knn_points, get_point_covariances, sample_farthest_points
from pytorch3d.loss import chamfer_distance
from .utils import *
from .mlp import get_mapping_mlp, TopkMLP
from .render import ProximityAttention
from utils import get_ray_sphere_intersection
from .render_self_attn import ProximitySelfAttention
from .unet import UNet
from .loss import get_loss

@torch.no_grad()
def _select_d2r_filter_chunked(rays_o, rays_d, points, k, eps, chunk_size=512):
    """Select forward-facing nearest-to-ray points without an all-rays tensor.

    The archived implementation materialized ``R x P x 3`` intermediates for
    every ray in a patch at once. Chunking only the independent ray dimension
    preserves its selection rule while bounding peak memory.
    """
    ray_shape = rays_d.shape[:-1]
    flat_origins = rays_o.reshape(-1, 3)
    flat_directions = rays_d.reshape(-1, 3)
    selected, minimum_distances = [], []
    for start in range(0, flat_directions.shape[0], chunk_size):
        stop = min(start + chunk_size, flat_directions.shape[0])
        origins = flat_origins[start:stop, None, :]
        directions = flat_directions[start:stop, None, :]
        vectors = points[None, :, :] - origins
        projection = directions * torch.sum(vectors * directions, dim=-1, keepdim=True)
        distances = torch.linalg.vector_norm(vectors - projection, dim=-1)
        minimum_distances.append(distances.min(dim=-1).values)
        cosine = torch.sum(
            normalize_vector(vectors, eps) * normalize_vector(directions, eps), dim=-1
        )
        distances.masked_fill_(cosine <= 0, 1e10)
        selected.append(torch.topk(distances, k, dim=-1, largest=False, sorted=False).indices)
    return (
        torch.cat(selected, dim=0).reshape(*ray_shape, k),
        torch.cat(minimum_distances, dim=0).reshape(*ray_shape),
    )


class PAPR(nn.Module):    
    def __init__(self, args, device='cuda'):
        super(PAPR, self).__init__()
        self.args = args
        self.eps = args.eps
        self.device = device

        self.use_amp = args.use_amp
        self.amp_dtype = torch.float16 if args.amp_dtype == 'float16' else torch.bfloat16
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)
        
        self.pruned_points = False
        self.added_points = False
        self.kept_indices = None

        point_opt = args.geoms.points
        pc_feat_opt = args.geoms.point_feats
        bkg_feat_opt = args.geoms.background
        exposure_opt = args.exposure_control
        self.exposure_opt = exposure_opt
        self.points_sampled = None
        
        self.pixel_frustum_margin = args.geoms.points.pixel_frustum_margin
        self._benchmark_knn_frustum = False  # Set to True to enable timing prints for knn_frustum
        
        # self.register_buffer('select_k', torch.tensor(point_opt.select_k, device=device, dtype=torch.int32))
        self.select_k = point_opt.select_k
        
        # self.knn_selector = NearestNeighbors(n_neighbors=self.select_k)
        self.pixel_width = None
        self.pixel_height = None

        self.coord_scale = args.dataset.coord_scale
        # Unified background sphere config for ray-sphere intersection
        self.bkg_sphere_radius = getattr(args.geoms.points, 'bkg_sphere_radius', 5.0)
        self.bkg_sphere_center = getattr(args.geoms.points, 'bkg_sphere_center', [0.0, 0.0, 0.0])
        self.gt_points = None
        if point_opt.load_path:
            if point_opt.load_path.endswith('.pth') or point_opt.load_path.endswith('.pt'):
                points = torch.load(point_opt.load_path, map_location='cpu', weights_only=True)
                points = np.asarray(points).astype(np.float32)
            elif point_opt.load_path.endswith('.ply'):
                pcd = o3d.io.read_point_cloud(point_opt.load_path)
                points = np.asarray(pcd.points).astype(np.float32)
            points = torch.from_numpy(points)
            print("Loaded points from {}, shape: {}, dtype {}".format(point_opt.load_path, points.shape, points.dtype))
            print("Loaded points scale: ", points[:, 0].min(), points[:, 0].max(), points[:, 1].min(), points[:, 1].max(), points[:, 2].min(), points[:, 2].max())
            
            # Optionally prune points outside the background sphere
            prune_outside_bkg_sphere = getattr(point_opt, 'prune_outside_bkg_sphere', False)
            if prune_outside_bkg_sphere:
                # Note: Points are still in canonical space here (before scaling on line 93)
                # so use canonical sphere params without coord_scale multiplication
                sphere_center = torch.tensor(self.bkg_sphere_center, dtype=points.dtype)
                sphere_radius = self.bkg_sphere_radius
                distances = torch.norm(points - sphere_center, dim=1)
                inside_mask = distances <= sphere_radius
                num_before = points.shape[0]
                points = points[inside_mask]
                num_after = points.shape[0]
                print("Pruned points outside bkg sphere (radius={:.3f}, center={}): {} -> {} ({} removed)".format(
                    sphere_radius, self.bkg_sphere_center, num_before, num_after, num_before - num_after))

            perm = torch.randperm(points.shape[0])
            selected_indices = perm[:args.max_num_pts]
            points = points[selected_indices, :].float()
            points = points * self.coord_scale
            
            pt_init_center = points.mean(dim=0)
            self.gt_points = points
        else:
            # Initialize point positions
            pt_init_center = [i * self.coord_scale for i in point_opt.init_center]
            pt_init_scale = [i * self.coord_scale for i in point_opt.init_scale]
            if point_opt.init_type == 'sphere': # initial points on a sphere
                points = sphere_pc(pt_init_center, point_opt.init_num, pt_init_scale)
            elif point_opt.init_type == 'cube': # initial points in a cube
                points = cube_normal_pc(pt_init_center, point_opt.init_num, pt_init_scale)
            elif point_opt.init_type == 'cube_uniform': # initial points in a cube uniformly
                pt_init_scale = torch.tensor(pt_init_scale, device=device)
                pt_init_center = torch.tensor(pt_init_center, device=device)
                points = (torch.rand(point_opt.init_num, 3, device=device) * 2 - 1) * pt_init_scale + pt_init_center
            else:
                raise NotImplementedError("Point init type [{:s}] is not found".format(point_opt.init_type))
            print("Initialized points scale: ", points[:, 0].min(), points[:, 0].max(), points[:, 1].min(), points[:, 1].max(), points[:, 2].min(), points[:, 2].max())
        
        # Append sphere background points to self.points if no_additional_bkg is True
        if getattr(self.args.geoms, 'no_additional_bkg', False) and getattr(self.args.geoms, 'append_sphere_bkg_points', False):
            sphere_bkg_num = getattr(self.args.geoms, 'sphere_bkg_points_num', 1000)
            # Reuse bkg_sphere_radius and bkg_sphere_center from geoms.points
            sphere_radius = point_opt.bkg_sphere_radius * self.coord_scale
            sphere_center = [c * self.coord_scale for c in point_opt.bkg_sphere_center]
            sphere_bkg_scale = [sphere_radius, sphere_radius, sphere_radius]
            sphere_bkg_points = sphere_pc(sphere_center, sphere_bkg_num, sphere_bkg_scale)
            print(f"Appending {sphere_bkg_num} sphere background points to self.points (no_additional_bkg mode)")
            print("Sphere bkg points scale: ", sphere_bkg_points[:, 0].min(), sphere_bkg_points[:, 0].max(), 
                  sphere_bkg_points[:, 1].min(), sphere_bkg_points[:, 1].max(), 
                  sphere_bkg_points[:, 2].min(), sphere_bkg_points[:, 2].max())
            points = torch.cat([points, sphere_bkg_points], dim=0)
            print(f"Total points after appending: {points.shape[0]}")
        
        self.points = nn.Parameter(points, requires_grad=True)

        self.bkg_points_pc_feats = None
        self.bkg_points_pc_feats_mlp = None
        self.bkg_points_pc_feats_encoder = None
        self.bkg_points_influ_scores = None
        self.bkg_points = None
        self.bkg_points_embedv = None
        if point_opt.use_bkg_points:
            assert not bkg_feat_opt.learn_bkg_token, "Background token should not be learned if using background points"
            assert "filter" in self.args.geoms.points.select_k_type or "frustum" in self.args.geoms.points.select_k_type, "Filtering should be used if using background points"
            sphere_center = [c * self.coord_scale for c in point_opt.bkg_sphere_center]
            sphere_radius = point_opt.bkg_sphere_radius * self.coord_scale
            bkg_points = sphere_pc(sphere_center, point_opt.bkg_points_num, [sphere_radius, sphere_radius, sphere_radius])
            self.bkg_points = nn.Parameter(bkg_points, requires_grad=False)
            print("Initialized bkg points scale: ", bkg_points[:, 0].min(), bkg_points[:, 0].max(), bkg_points[:, 1].min(), bkg_points[:, 1].max(), bkg_points[:, 2].min(), bkg_points[:, 2].max())

            self.bkg_points_influ_scores = nn.Parameter(torch.ones(1, 1, device=device) * point_opt.bkg_points_influ_init_val, requires_grad=point_opt.bkg_points_influ_learn)
            # Use MLP to generate features from background color instead of learned feature vectors
            use_bkg_points_pc_feats_mlp = getattr(point_opt, 'use_bkg_points_pc_feats_mlp', False)
            if use_bkg_points_pc_feats_mlp:
                # Add positional encoding layer if config provided
                self.bkg_points_pc_feats_encoder = None
                if hasattr(point_opt, 'bkg_points_pc_feats_encode_config') and point_opt.bkg_points_pc_feats_encode_config is not None:
                    use_tcnn_encoder = args.models.attn.use_tcnn_encoder if hasattr(args.models.attn, 'use_tcnn_encoder') else True
                    self.bkg_points_pc_feats_encoder = Encoding(3, point_opt.bkg_points_pc_feats_encode_config, use_tcnn_encoder)
                    mlp_input_dim = self.bkg_points_pc_feats_encoder.n_output_dims
                    print("Using encoding for bkg_points_pc_feats: input_dim=3, encoded_dim={}".format(mlp_input_dim))
                else:
                    mlp_input_dim = 3
                
                # Input is encoded background color (or raw if no encoding), output is pc_feat_opt.dim
                assert hasattr(point_opt, 'bkg_points_pc_feats_mlp_config') and point_opt.bkg_points_pc_feats_mlp_config is not None, \
                    "bkg_points_pc_feats_mlp_config must be provided when use_bkg_points_pc_feats_mlp is True"
                self.bkg_points_pc_feats_mlp = tcnn.Network(mlp_input_dim, pc_feat_opt.dim, point_opt.bkg_points_pc_feats_mlp_config)
                self.bkg_points_pc_feats_mlp.params.data[...] = get_tcnn_init_weights(mlp_input_dim, pc_feat_opt.dim, point_opt.bkg_points_pc_feats_mlp_config, device=self.bkg_points_pc_feats_mlp.params.device)
                self.bkg_points_pc_feats = None
                print("Using MLP for bkg_points_pc_feats: mlp_input_dim={}, output_dim={}".format(mlp_input_dim, pc_feat_opt.dim))
            else:
                # Fallback to old method: learned feature vectors
                num_feats = 4 if args.rnd_background and args.rnd_background_use_4_feats else 1
                self.bkg_points_pc_feats = nn.Parameter(torch.randn(num_feats, pc_feat_opt.dim), requires_grad=True)
                self.bkg_points_pc_feats_mlp = None
                self.bkg_points_pc_feats_encoder = None
            
            if point_opt.bkg_points_embedv_init_type == "randn":
                self.bkg_points_embedv = nn.Parameter(torch.randn(args.models.attn.output_dim_v, device=device), requires_grad=point_opt.bkg_points_embedv_learn)
            elif point_opt.bkg_points_embedv_init_type == "zeros":
                self.bkg_points_embedv = nn.Parameter(torch.zeros(args.models.attn.output_dim_v, device=device), requires_grad=point_opt.bkg_points_embedv_learn)
            elif point_opt.bkg_points_embedv_init_type == "ones":
                self.bkg_points_embedv = nn.Parameter(torch.ones(args.models.attn.output_dim_v, device=device), requires_grad=point_opt.bkg_points_embedv_learn)
            else:
                raise NotImplementedError("Bkg points embedv init type [{:s}] is not found".format(point_opt.bkg_points_embedv_init_type))
            print("Initialized bkg points embedv: ", self.bkg_points_embedv.shape, self.bkg_points_embedv.min().item(), self.bkg_points_embedv.max().item(), "learnable: ", point_opt.bkg_points_embedv_learn)

        # Initialize point influence scores
        if point_opt.influ_init_func == 'ones':
            self.points_influ_scores = nn.Parameter(torch.ones(points.shape[0], 1, device=device) * point_opt.influ_init_val, requires_grad=True)
        elif point_opt.influ_init_func == 'randn':
            self.points_influ_scores = nn.Parameter(torch.randn(points.shape[0], 1, device=device) * point_opt.influ_init_val, requires_grad=True)
        elif point_opt.influ_init_func == 'rand':
            self.points_influ_scores = nn.Parameter(torch.rand(points.shape[0], 1, device=device) + point_opt.influ_init_val, requires_grad=True)
        self.points_influ_scores_act = activation_func(point_opt.influ_act)
        
        # Initialize point scaler
        scaler_init_val = getattr(point_opt, 'scaler_init_val', 1.0)
        scaler_learn = getattr(point_opt, 'scaler_learn', False)
        if getattr(point_opt, 'scaler_init_func', 'ones') == 'ones':
            self.points_scaler = nn.Parameter(torch.ones(points.shape[0], 1, device=device) * scaler_init_val, requires_grad=scaler_learn)
        elif getattr(point_opt, 'scaler_init_func', 'ones') == 'randn':
            self.points_scaler = nn.Parameter(torch.randn(points.shape[0], 1, device=device) * scaler_init_val, requires_grad=scaler_learn)
        elif getattr(point_opt, 'scaler_init_func', 'ones') == 'rand':
            self.points_scaler = nn.Parameter(torch.rand(points.shape[0], 1, device=device) * scaler_init_val + (1.0 - scaler_init_val), requires_grad=scaler_learn)
        else:
            self.points_scaler = nn.Parameter(torch.ones(points.shape[0], 1, device=device) * scaler_init_val, requires_grad=scaler_learn)
        print("Point scaler: ", self.points_scaler.shape, self.points_scaler.min().item(), self.points_scaler.max().item(), self.points_scaler.mean().item(), "learnable: ", scaler_learn)
        
        if args.geoms.alpha.use:
            if args.geoms.alpha.init_func == 'ones':
                alpha_init_func = torch.ones
            elif args.geoms.alpha.init_func == 'empty':
                alpha_init_func = torch.empty
            elif args.geoms.alpha.init_func == 'randn':
                alpha_init_func = torch.randn
            else:
                raise NotImplementedError

            if args.geoms.alpha.act == 'sigmoid':
                self.inverse_alpha_activation = inverse_sigmoid
            elif args.geoms.alpha.act == 'none':
                self.inverse_alpha_activation = nn.Identity()
            else:
                raise NotImplementedError

            self.points_alpha = nn.Parameter(self.inverse_alpha_activation(alpha_init_func(points.shape[0], 1, device=device) * args.geoms.alpha.init_val), requires_grad=True)
            self.points_alpha_act = activation_func(args.geoms.alpha.act)
                
            self.alpha_attn_loss = get_loss(args.geoms.alpha.losses)
        else:
            self.points_alpha = None
            self.points_alpha_act = None
            self.alpha_attn_loss = None
            
        self.points_last_grad = nn.Parameter(torch.zeros(points.shape[0], 3, device=device), requires_grad=False)
        self.points_acc_grad = nn.Parameter(torch.zeros(points.shape[0], 3, device=device), requires_grad=False)
        self.points_acc_grad_norm = nn.Parameter(torch.zeros(points.shape[0], device=device), requires_grad=False)
        self.points_grad_cnt = nn.Parameter(torch.zeros(points.shape[0], device=device), requires_grad=False)
        
        # Point density estimation parameters
        self.density_k = getattr(point_opt, 'density_k', 30)  # Number of neighbors for density estimation
        self.density_update_interval = getattr(point_opt, 'density_update_interval', 1000)  # Update density every T iterations
        self.grad_noise_std = getattr(point_opt, 'grad_noise_std', 0.0)  # Std of noise perturbation for gradients (scaled by 1/density)
        self.points_density = nn.Parameter(torch.ones(points.shape[0], device=device), requires_grad=False)
        # Initialize point density
        self.update_point_density()

        # Initialize mapping MLP, only if fine-tuning with IMLE for the exposure control
        self.mapping_mlp = None
        if exposure_opt.use:
            self.mapping_mlp = get_mapping_mlp(exposure_opt, use_amp=self.use_amp, amp_dtype=self.amp_dtype)

        self.topk_mlp = None
        if "mlp" in self.args.geoms.points.select_k_type:
            if self.args.geoms.points.project:
                num_feats_dict = {
                    "mlp1": 2,
                    "mlp2": 2,
                    "mlp3": 2,
                    "mlp4": 3,
                    "mlp5": 3,
                    "mlp6": 4,
                }
            else:
                num_feats_dict = {
                    "mlp1": 2,
                    "mlp2": 2,
                    "mlp3": 2,
                    "mlp4": 3,
                    "mlp5": 5,
                    "mlp6": 3,
                    "mlp7": 7,
                    "mlp8": 6,
                }
            num_feats = num_feats_dict[self.args.geoms.points.select_k_type]
            self.topk_mlp = TopkMLP(args.models.topk_mlp, num_feats, device=device, use_amp=self.use_amp, amp_dtype=self.amp_dtype)
        
        # Initialize UNet
        self.unet = None
        self.fused_feature_mlp = None
        self.feat_map_dim = args.models.attn.output_dim_v
        if args.models.unet.use:
            unet_groups = 1
            unet_output_dim = 3
            self.fused_feature_encoder = Encoding(args.models.attn.output_dim_v, args.models.unet.encode_config, True)
            unet_input_dim = self.fused_feature_encoder.n_output_dims
            if self.args.models.unet.double_channel:
                unet_input_dim *= 2
                print("Doubling unet input dim to ", unet_input_dim)
            self.unet_input_dim = unet_input_dim
            self.unet = UNet(n_channels=unet_input_dim, \
                             n_classes=unet_output_dim, bilinear=args.models.unet.bilinear, \
                             single=args.models.unet.single, norm=args.models.unet.norm, \
                             last_act=args.models.unet.last_act, act=args.models.unet.act, \
                             inp_scale=args.models.unet.inp_scale, use_outc=args.models.unet.use_outc, \
                             affine_layer=args.models.unet.affine_layer, channel_factor=args.models.unet.channel_factor, \
                             use_amp=self.use_amp, amp_dtype=self.amp_dtype, groups=unet_groups, group_last=args.models.unet.group_last)
            print("Number of parameters of unet: ", count_parameters(self.unet))
        elif args.models.fused_feature_mlp.use:
            self.fused_feature_encoder = Encoding(args.models.attn.output_dim_v, args.models.fused_feature_mlp.encode_config, True)
            self.fused_feature_mlp = tcnn.Network(self.fused_feature_encoder.n_output_dims, 3, args.models.fused_feature_mlp.mlp_config)
        else:
            assert args.models.attn.output_dim_v == 3, \
                "Value embedding MLP should have output dim 3 if not using unet"

        self.texture_v_network = None
        if args.texture.use_v:
            self.texture_v_encoder = Encoding(self.proximity_attn.input_texture_dim_v, args.texture.encode_v_config, True)
            if args.texture.network_v:
                input_dim = self.texture_v_encoder.n_output_dims + args.texture.encode_additional_dim_v
                if args.texture.fuse_v_with_nuvo_texture_map:
                    input_dim += self.texture_pe_layer.n_output_dims
                if args.texture.v_type == 99:
                    input_dim = args.texture.encode_additional_dim_v
                self.texture_v_network = tcnn.Network(input_dim, args.texture.v_dim, args.texture.network_v_config)
                self.texture_v_network.params.data[...] = get_tcnn_init_weights(input_dim, args.texture.v_dim, args.texture.network_v_config, device=self.texture_v_network.params.device)
            print("Texture V Network: ", self.texture_v_network)
            
            self.texture_v_scheduler = lambda step: args.texture.v_weight * step / args.texture.v_weight_warmup_steps \
                                                        if args.texture.v_weight_warmup_steps > 0 and (step < args.texture.v_weight_warmup_steps) \
                                                            else args.texture.v_weight
        if args.texture.use_combine_network:
            network_inp_dim = self.feat_map_dim
            if args.texture.combine_network_input_both:
                network_inp_dim *= 2
            self.texture_v_network = tcnn.Network(network_inp_dim, self.feat_map_dim, args.texture.network_v_config)
            self.texture_v_network.params.data[...] = get_tcnn_init_weights(network_inp_dim, self.feat_map_dim, args.texture.network_v_config, device=self.texture_v_network.params.device)
            print("Texture V Network: ", self.texture_v_network)
                
        self.dumb_constant = 1 / (np.exp(bkg_feat_opt.constant) / (np.exp(bkg_feat_opt.constant) + point_opt.select_k))
        print("Dumb constant: ", self.dumb_constant)

        # Initialize background score and features
        self.bkg_feats = nn.Parameter(torch.FloatTensor(bkg_feat_opt.init_color)[None, :], requires_grad=bkg_feat_opt.learnable)
        bkg_score = torch.tensor(bkg_feat_opt.constant, device=device, dtype=torch.float32).reshape(1)
        self.bkg_score = nn.Parameter(bkg_score, requires_grad=bkg_feat_opt.learn_score)
        scaler_init_val = 1.0 if not args.rnd_background_use_dumb_constant else self.dumb_constant
        self.bkg_scaler = nn.Parameter(torch.ones_like(self.bkg_feats) * scaler_init_val, requires_grad=bkg_feat_opt.learn_scaler)
        
        # Initialize learnable cubemap textures for background
        self.cubemap_textures = None  # RGB cubemap for background color
        self.cubemap_feature_textures = None  # Feature cubemap for append_bkg_points_feats
        self.cubemap_activation = None
        self.cubemap_feature_activation = None
        self.cubemap_use_feature_map = False
        self.cubemap_use_rgb = False
        self.cubemap_feature_dim = None
        self.cubemap_start_step = 0
        cubemap_opt = getattr(args, 'cubemap_texture', None)
        if cubemap_opt is not None:
            # Check which cubemaps to create
            self.cubemap_use_rgb = getattr(cubemap_opt, 'use_rgb', False)
            self.cubemap_use_feature_map = getattr(cubemap_opt, 'use_feature_map', False)
            
            # Only proceed if at least one cubemap type is enabled
            if self.cubemap_use_rgb or self.cubemap_use_feature_map:
                resolution = getattr(cubemap_opt, 'resolution', 256)
                init_type = getattr(cubemap_opt, 'init_type', 'random')
                init_value = getattr(cubemap_opt, 'init_value', 1.0)
                self.cubemap_start_step = getattr(cubemap_opt, 'start_step', 0)
            
            # Create RGB cubemap if enabled
            if self.cubemap_use_rgb:
                # Get activation function config for RGB cubemap (default: sigmoid)
                cubemap_act_type = getattr(cubemap_opt, 'activation', 'sigmoid')
                self.cubemap_activation = activation_func(cubemap_act_type)
                
                # Create 6 face textures for RGB: [+X, -X, +Y, -Y, +Z, -Z]
                cubemap_list = []
                for i in range(6):
                    if init_type == "constant":
                        texture = torch.ones(3, resolution, resolution) * init_value
                    elif init_type == "random":
                        texture = torch.rand(3, resolution, resolution)
                    elif init_type == "randn":
                        texture = torch.randn(3, resolution, resolution) * 0.1 + 0.5
                    else:
                        raise NotImplementedError(f"Cubemap init type [{init_type}] is not found")
                    
                    # Apply inverse activation for proper initialization (only for bounded activations)
                    if cubemap_act_type == 'sigmoid':
                        texture = texture.clamp(1e-4, 1 - 1e-4)
                        texture = torch.log(texture / (1 - texture))
                    elif cubemap_act_type == 'tanh':
                        texture = texture.clamp(-1 + 1e-4, 1 - 1e-4)
                        texture = 0.5 * torch.log((1 + texture) / (1 - texture))
                    
                    cubemap_list.append(nn.Parameter(texture, requires_grad=True))
                self.cubemap_textures = nn.ParameterList(cubemap_list)
                print(f"Initialized cubemap RGB textures: 6 x (3, {resolution}, {resolution}), init_type: {init_type}, activation: {cubemap_act_type}, start_step: {self.cubemap_start_step}")
            
            # Create feature cubemap if enabled
            if self.cubemap_use_feature_map:
                # When using feature cubemap, alpha blending should be disabled
                # because the background RGB is computed from the weighted sum of embedv (last dimension)
                assert not getattr(args.models.attn, 'append_bkg_points_alpha_blend', False), \
                    "append_bkg_points_alpha_blend must be False when using cubemap_feature_textures. " \
                    "The background is handled through the appended virtual point's contribution to embedv."
                # Use specified feature_dim or default to pc_feat_opt.dim
                self.cubemap_feature_dim = getattr(cubemap_opt, 'feature_dim', None) or pc_feat_opt.dim
                feature_resolution = getattr(cubemap_opt, 'feature_resolution', None) or resolution
                feature_init_type = getattr(cubemap_opt, 'feature_init_type', None) or init_type
                feature_init_value = getattr(cubemap_opt, 'feature_init_value', 0.0)
                
                # Get activation for feature cubemap (default: none)
                feature_act_type = getattr(cubemap_opt, 'feature_activation', 'none')
                self.cubemap_feature_activation = activation_func(feature_act_type)
                
                # Create 6 face textures for features
                feature_cubemap_list = []
                for i in range(6):
                    if feature_init_type == "constant":
                        texture = torch.ones(self.cubemap_feature_dim, feature_resolution, feature_resolution) * feature_init_value
                    elif feature_init_type == "random":
                        texture = torch.rand(self.cubemap_feature_dim, feature_resolution, feature_resolution)
                    elif feature_init_type == "randn":
                        texture = torch.randn(self.cubemap_feature_dim, feature_resolution, feature_resolution) * feature_init_value
                    else:
                        raise NotImplementedError(f"Cubemap feature init type [{feature_init_type}] is not found")
                    
                    # Apply inverse activation for proper initialization (only for bounded activations)
                    if feature_act_type == 'sigmoid':
                        texture = texture.clamp(1e-4, 1 - 1e-4)
                        texture = torch.log(texture / (1 - texture))
                    elif feature_act_type == 'tanh':
                        texture = texture.clamp(-1 + 1e-4, 1 - 1e-4)
                        texture = 0.5 * torch.log((1 + texture) / (1 - texture))
                    
                    feature_cubemap_list.append(nn.Parameter(texture, requires_grad=True))
                self.cubemap_feature_textures = nn.ParameterList(feature_cubemap_list)
                print(f"Initialized cubemap feature textures: 6 x ({self.cubemap_feature_dim}, {feature_resolution}, {feature_resolution}), init_type: {feature_init_type}, activation: {feature_act_type}")
        
        # Initialize learnable cubemap depth textures for background surface offset
        self.cubemap_depth_textures = None  # Depth offset cubemap (1 channel per face)
        self.cubemap_depth_scale = 1.0  # Scale factor for depth offset
        self.cubemap_use_depth_map = False
        if cubemap_opt is not None:
            self.cubemap_use_depth_map = getattr(cubemap_opt, 'use_depth_map', False)
            
            if self.cubemap_use_depth_map:
                depth_resolution = getattr(cubemap_opt, 'depth_resolution', None) or getattr(cubemap_opt, 'resolution', 256)
                depth_init_type = getattr(cubemap_opt, 'depth_init_type', 'constant')
                depth_init_value = getattr(cubemap_opt, 'depth_init_value', 0.0)  # Default: no offset
                self.cubemap_depth_scale = getattr(cubemap_opt, 'depth_scale', 1.0)
                
                # Create 6 face textures for depth: [+X, -X, +Y, -Y, +Z, -Z]
                # No activation is used for depth - values are unbounded offsets
                depth_cubemap_list = []
                for i in range(6):
                    if depth_init_type == "constant":
                        texture = torch.ones(1, depth_resolution, depth_resolution) * depth_init_value
                    elif depth_init_type == "random":
                        texture = torch.rand(1, depth_resolution, depth_resolution) * 0.1  # Small random init
                    elif depth_init_type == "randn":
                        texture = torch.randn(1, depth_resolution, depth_resolution) * 0.01  # Very small random init
                    else:
                        raise NotImplementedError(f"Cubemap depth init type [{depth_init_type}] is not found")
                    
                    depth_cubemap_list.append(nn.Parameter(texture, requires_grad=True))
                self.cubemap_depth_textures = nn.ParameterList(depth_cubemap_list)
                print(f"Initialized cubemap depth textures: 6 x (1, {depth_resolution}, {depth_resolution}), init_type: {depth_init_type}, scale: {self.cubemap_depth_scale}")
        
        self.bkg_token = None
        if bkg_feat_opt.learn_bkg_token:
            self.bkg_token = nn.Parameter(torch.randn(args.models.attn.output_dim_k, device=device), requires_grad=True)

        # Initialize appended background point embedding
        self.append_bkg_points_embedv = None
        self.append_bkg_points_feats = None
        if args.models.attn.append_bkg_points:
            num_feats = 4 if args.rnd_background and args.rnd_background_use_4_feats else 1
            self.append_bkg_points_feats = nn.Parameter(torch.randn(num_feats, pc_feat_opt.dim), requires_grad=True)
            if getattr(args.models.attn, 'append_bkg_points_use_embedv', False):
                embedv_init_type = getattr(args.models.attn, 'append_bkg_points_embedv_init_type', 'randn')
                embedv_learn = getattr(args.models.attn, 'append_bkg_points_embedv_learn', True)
                if embedv_init_type == "randn":
                    self.append_bkg_points_embedv = nn.Parameter(torch.randn(num_feats, args.models.attn.output_dim_v, device=device), requires_grad=embedv_learn)
                elif embedv_init_type == "zeros":
                    self.append_bkg_points_embedv = nn.Parameter(torch.zeros(num_feats, args.models.attn.output_dim_v, device=device), requires_grad=embedv_learn)
                elif embedv_init_type == "ones":
                    self.append_bkg_points_embedv = nn.Parameter(torch.ones(num_feats, args.models.attn.output_dim_v, device=device), requires_grad=embedv_learn)
                else:
                    raise NotImplementedError("Append bkg points embedv init type [{:s}] is not found".format(embedv_init_type))
                print("Initialized append bkg points embedv: ", self.append_bkg_points_embedv.shape, self.append_bkg_points_embedv.min().item(), self.append_bkg_points_embedv.max().item(), "learnable: ", embedv_learn)

        # Initialize learnable background exp scaler for attention
        self.bkg_exp_scaler = None
        if args.models.attn.append_bkg_points and getattr(args.models.attn, 'bkg_exp_scaler', False):
            bkg_exp_scaler_init = getattr(args.models.attn, 'bkg_exp_scaler_init', 1.0)
            self.bkg_exp_scaler = nn.Parameter(torch.tensor([bkg_exp_scaler_init], device=device), requires_grad=True)
            print(f"Initialized bkg_exp_scaler: {self.bkg_exp_scaler.item():.4f}")

        # Initialize point features
        if args.models.attn.use_pc_feats_directly and (pc_feat_opt.dim == 3 or args.models.attn.use_sh):
            if args.models.attn.use_sh:
                if pc_feat_opt.dim == 3 or pc_feat_opt.dim // (args.models.attn.sh_degree + 1) ** 2 == 3:
                    colors = torch.zeros(points.shape[0], (args.models.attn.sh_degree + 1) ** 2, 3)
                    colors[:, 0, :] = rgb_to_sh(torch.rand(points.shape[0], 3))
                    self.pc_feats = nn.Parameter(colors, requires_grad=True)
                else:
                    num_bases = (args.models.attn.sh_degree + 1) ** 2
                    self.pc_feats = nn.Parameter(torch.rand(points.shape[0], num_bases, pc_feat_opt.dim // num_bases), requires_grad=True)
            else:
                self.pc_feats = nn.Parameter(torch.rand(points.shape[0], pc_feat_opt.dim), requires_grad=True)
        else:
            self.pc_feats = nn.Parameter(torch.randn(points.shape[0], pc_feat_opt.dim), requires_grad=True)
        print("Point features: ", self.pc_feats.shape, self.pc_feats.min().item(), self.pc_feats.max().item(), self.pc_feats.mean().item(), self.pc_feats.std().item())

        self.last_act = activation_func(args.models.last_act)
        self.bkg_attn_act = activation_func(args.models.bkg_attn_act)

        # Initialize proximity attention layer
        if args.models.attn_type == "proximity":
            # Create a modified args.models.attn with unified sphere config
            attn_args = args.models.attn
            attn_args.bkg_sphere_radius = self.bkg_sphere_radius
            attn_args.bkg_sphere_center = self.bkg_sphere_center
            self.proximity_attn = ProximityAttention(attn_args, point_feats_dim=pc_feat_opt.dim, use_amp=self.use_amp, amp_dtype=self.amp_dtype, attn_act=args.attn_act, attn_act_temp=args.attn_act_temp, coord_scale=self.coord_scale)
        elif args.models.attn_type == "proximity_self":
            self.proximity_attn = ProximitySelfAttention(args.models.self_attn, point_feats_dim=pc_feat_opt.dim)
        else:
            raise NotImplementedError("Unknown proximity attention type: {}".format(args.models.attn_type))

        self.init_optimizers(total_steps=0)


    def init_optimizers(self, total_steps):
        self.optimizers = {}
        self.schedulers = {}
        lr_opt = self.args.training.lr
        betas = (lr_opt.beta1, lr_opt.beta2)
        debug = False
        use_warmup = lr_opt.use_warmup
        print("use_warmup: ", use_warmup)
        param_list = {
            "points": [self.points, lr_opt.points],
            "points_influ_scores": [self.points_influ_scores, lr_opt.points_influ_scores],
            "points_scaler": [self.points_scaler, getattr(lr_opt, 'points_scaler', lr_opt.points_influ_scores)],
            "pc_feats": [self.pc_feats, lr_opt.feats],
            "points_alpha": [self.points_alpha, lr_opt.points_alpha],
            "bkg_feats": [self.bkg_feats, lr_opt.bkg_feats],
            "bkg_token": [self.bkg_token, lr_opt.bkg_token],
            "unet": [self.unet, lr_opt.unet],
            "mapping_mlp": [self.mapping_mlp, lr_opt.mapping_mlp],
            "topk_mlp": [self.topk_mlp, lr_opt.topk_mlp],
            "bkg_scaler": [self.bkg_scaler, lr_opt.bkg_scaler],
            "bkg_score": [self.bkg_score, lr_opt.bkg_score],
            "bkg_points_pc_feats": [self.bkg_points_pc_feats, lr_opt.bkg_points_pc_feats],
            "bkg_points_pc_feats_mlp": [self.bkg_points_pc_feats_mlp, lr_opt.bkg_points_pc_feats],
            "bkg_points_influ_scores": [self.bkg_points_influ_scores, lr_opt.bkg_points_influ_scores],
            "bkg_points": [self.bkg_points, lr_opt.bkg_points],
            "bkg_points_embedv": [self.bkg_points_embedv, lr_opt.bkg_points_embedv],
            "append_bkg_points_embedv": [self.append_bkg_points_embedv, lr_opt.bkg_points_embedv],
            "append_bkg_points_feats": [self.append_bkg_points_feats, lr_opt.bkg_points_pc_feats],
            "bkg_exp_scaler": [self.bkg_exp_scaler, getattr(lr_opt, 'bkg_exp_scaler', lr_opt.bkg_scaler)],
            "fused_feature_mlp": [self.fused_feature_mlp, lr_opt.fused_feature_mlp],
            "cubemap_textures": [self.cubemap_textures, lr_opt.cubemap_textures],
            "cubemap_feature_textures": [self.cubemap_feature_textures, lr_opt.cubemap_textures],
            "cubemap_depth_textures": [self.cubemap_depth_textures, lr_opt.cubemap_depth_textures],
        }
        for name, param_config in param_list.items():
            param, param_lr_opt = param_config
            require_grad = param.requires_grad if isinstance(param, torch.Tensor) else True
            if param is not None and require_grad:
                opt_params = [param] if isinstance(param, torch.Tensor) else param.parameters()
                if lr_opt.optim_type == 'Adam':
                    self.optimizers[name] = torch.optim.Adam(opt_params, lr=param_lr_opt.base_lr * lr_opt.lr_factor, weight_decay=param_lr_opt.weight_decay, eps=lr_opt.eps, amsgrad=lr_opt.use_amsgrad, betas=betas)
                    self.schedulers[name] = create_learning_rate_fn(self.optimizers[name], self.args.training.steps, param_lr_opt, use_warmup=lr_opt.use_warmup, debug=debug)
                elif lr_opt.optim_type == 'AdamW':
                    self.optimizers[name] = torch.optim.AdamW(opt_params, lr=param_lr_opt.base_lr * lr_opt.lr_factor, weight_decay=param_lr_opt.weight_decay, eps=lr_opt.eps, amsgrad=lr_opt.use_amsgrad, betas=betas)
                    self.schedulers[name] = create_learning_rate_fn(self.optimizers[name], self.args.training.steps, param_lr_opt, use_warmup=lr_opt.use_warmup, debug=debug)
                else:
                    raise ValueError("Unknown optimizer type: {}".format(lr_opt.optim_type))

        # Create separate optimizers for proximity_attn: attn_v and attn_other
        if self.proximity_attn is not None and hasattr(self.proximity_attn, 'model_v') and self.proximity_attn.model_v is not None:
            param_lr_opt = lr_opt.attn
            
            # Get parameters from model_v for attn_v optimizer
            attn_v_params = list(self.proximity_attn.model_v.parameters())
            
            # Get all other parameters from proximity_attn (excluding model_v)
            attn_other_params = []
            model_v_param_ids = set(id(p) for p in attn_v_params)
            for name, param in self.proximity_attn.named_parameters():
                if id(param) not in model_v_param_ids:
                    attn_other_params.append(param)
            
            # Create attn_v optimizer
            if attn_v_params:
                if lr_opt.optim_type == 'Adam':
                    self.optimizers["attn_v"] = torch.optim.Adam(attn_v_params, lr=param_lr_opt.base_lr * lr_opt.lr_factor, weight_decay=param_lr_opt.weight_decay, eps=lr_opt.eps, amsgrad=lr_opt.use_amsgrad, betas=betas)
                    self.schedulers["attn_v"] = create_learning_rate_fn(self.optimizers["attn_v"], self.args.training.steps, param_lr_opt, use_warmup=lr_opt.use_warmup, debug=debug)
                elif lr_opt.optim_type == 'AdamW':
                    self.optimizers["attn_v"] = torch.optim.AdamW(attn_v_params, lr=param_lr_opt.base_lr * lr_opt.lr_factor, weight_decay=param_lr_opt.weight_decay, eps=lr_opt.eps, amsgrad=lr_opt.use_amsgrad, betas=betas)
                    self.schedulers["attn_v"] = create_learning_rate_fn(self.optimizers["attn_v"], self.args.training.steps, param_lr_opt, use_warmup=lr_opt.use_warmup, debug=debug)

            # Create attn_other optimizer
            if attn_other_params:
                if lr_opt.optim_type == 'Adam':
                    self.optimizers["attn_other"] = torch.optim.Adam(attn_other_params, lr=param_lr_opt.base_lr * lr_opt.lr_factor, weight_decay=param_lr_opt.weight_decay, eps=lr_opt.eps, amsgrad=lr_opt.use_amsgrad, betas=betas)
                    self.schedulers["attn_other"] = create_learning_rate_fn(self.optimizers["attn_other"], self.args.training.steps, param_lr_opt, use_warmup=lr_opt.use_warmup, debug=debug)
                elif lr_opt.optim_type == 'AdamW':
                    self.optimizers["attn_other"] = torch.optim.AdamW(attn_other_params, lr=param_lr_opt.base_lr * lr_opt.lr_factor, weight_decay=param_lr_opt.weight_decay, eps=lr_opt.eps, amsgrad=lr_opt.use_amsgrad, betas=betas)
                    self.schedulers["attn_other"] = create_learning_rate_fn(self.optimizers["attn_other"], self.args.training.steps, param_lr_opt, use_warmup=lr_opt.use_warmup, debug=debug)
        for name in self.args.training.fix_keys:
            if name in self.optimizers:
                print("Fixing {}".format(name))
                self.optimizers.pop(name)
                self.schedulers.pop(name)

            # Handle attn -> attn_v and attn_other mapping for backward compatibility
            if name == 'attn':
                if "attn_v" in self.optimizers:
                    print("Fixing attn_v")
                    self.optimizers.pop("attn_v")
                    self.schedulers.pop("attn_v")
                if "attn_other" in self.optimizers:
                    print("Fixing attn_other")
                    self.optimizers.pop("attn_other")
                    self.schedulers.pop("attn_other")

        print(self.optimizers.keys())
        print(self.schedulers.keys())

        if total_steps > 0:
            for _, scheduler in self.schedulers.items():
                if scheduler is not None:
                    for _ in range(total_steps):
                        scheduler.step()


    def clear_optimizer(self):
        self.optimizers.clear()
        del self.optimizers


    def clear_scheduler(self):
        self.schedulers.clear()
        del self.schedulers


    def clear_grad(self):
        for _, optimizer in self.optimizers.items():
            if optimizer is not None:
                optimizer.zero_grad()


    @torch.no_grad()
    def perturb_rays(self, rays_o, pixels, c2w, pix_coords):
        N = rays_o.shape[0]
        # print(rays_o.shape, pixels.shape, c2w.shape, pix_coords.shape, pixels[0, 0:2, 0:2, :], pixels[1, 0:2, 0:2, :])
        # exit(0)
        # pixels = world_to_cam(rays_o + rays_d, c2w, vector=False)  # (N, H, W, 3)
        x = pixels[..., 0]
        y = pixels[..., 1]
        if self.pixel_width is None or self.pixel_height is None:   # Assuming single camera intrinsics and resolution
            self.pixel_width = (torch.roll(x, shifts=1, dims=2) - x)[:, :, 1:].abs().mean()
            self.pixel_height = (torch.roll(y, shifts=1, dims=1) - y)[:, 1:, :].abs().mean()
            print("pixel_width", self.pixel_width, "pixel_height", self.pixel_height)
        min_x = x.view(N, -1).min(dim=-1)[0][:, None, None]
        max_x = x.view(N, -1).max(dim=-1)[0][:, None, None]
        min_y = y.view(N, -1).min(dim=-1)[0][:, None, None]
        max_y = y.view(N, -1).max(dim=-1)[0][:, None, None]
        # print("self.pixel_width", self.pixel_width.shape, self.pixel_width.min(), self.pixel_width.max())
        # print("self.pixel_height", self.pixel_height.shape, self.pixel_height.min(), self.pixel_height.max())
        # print("min_x", min_x, "max_x", max_x, "min_y", min_y, "max_y", max_y)
        if self.args.geoms.rays.perturb_noise == "uniform":
            perturb_x = torch.empty_like(x).uniform_(-self.pixel_width / 2, self.pixel_width / 2)
            perturb_y = torch.empty_like(y).uniform_(-self.pixel_height / 2, self.pixel_height / 2)
        elif self.args.geoms.rays.perturb_noise == "normal":
            sigma_x = (self.pixel_width / 4) * self.args.geoms.rays.perturb_sigma_factor # 95% of the noises are within 2 sigma
            sigma_y = (self.pixel_height / 4) * self.args.geoms.rays.perturb_sigma_factor
            perturb_x = (torch.randn_like(x) * sigma_x).clamp_(min=-(self.pixel_width / 2), max=(self.pixel_width / 2))
            perturb_y = (torch.randn_like(y) * sigma_y).clamp_(min=-(self.pixel_height / 2), max=(self.pixel_height / 2))
        else:
            raise NotImplementedError("Unknown perturb noise type: {}".format(self.args.geoms.rays.perturb_noise))
        # print(perturb_x.shape, perturb_x.min(), perturb_x.max())
        # print(perturb_y.shape, perturb_y.min(), perturb_y.max())
        perturbed_pixels = torch.stack([(x + perturb_x).clamp_(min=min_x, max=max_x), (y + perturb_y).clamp_(min=min_y, max=max_y), pixels[..., 2]], dim=-1)
        # print(perturbed_pixels[..., 0].min(), perturbed_pixels[..., 0].max())
        # print(perturbed_pixels[..., 1].min(), perturbed_pixels[..., 1].max())
        grid = torch.stack([(perturbed_pixels[..., 0] - min_x) / (max_x - min_x) * 2 - 1,
                            -((perturbed_pixels[..., 1] - min_y) / (max_y - min_y) * 2 - 1)], dim=-1)
        # print(grid.shape, grid.min(), grid.max())
        if rays_o.dim() == 2:
            rays_o_expanded = rays_o[:, None, None, :]
        elif rays_o.dim() == 4:
            rays_o_expanded = rays_o
        else:
            raise ValueError(f"Unsupported rays_o ndim={rays_o.dim()}, expected 2 or 4")
        perturbed_rays_d = cam_to_world(perturbed_pixels, c2w, vector=False) - rays_o_expanded
        perturbed_rays_d = normalize_vector(perturbed_rays_d)
        # print(rays_d[0, 0, 0], perturbed_rays_d[0, 0, 0])
        # print(rays_d[0, 0, 1], perturbed_rays_d[0, 0, 1])
        # print(pix_coords.shape, grid.shape, pix_coords.dtype, grid.dtype)
        pix_coords = nn.functional.grid_sample(pix_coords.permute(0, 3, 1, 2), grid, mode='bilinear', align_corners=True).permute(0, 2, 3, 1)
        return perturbed_rays_d, grid, pix_coords


    def get_points(self, deformed_points=None, drop=False, bkg_points_scale=1.0, cur_step=-1):
        """
        Get points, optionally with deformed_points and/or bkg_points concatenated.
        
        Args:
            deformed_points: Optional tensor of deformed points to use instead of self.points
            drop: Whether to apply dropout to points
            bkg_points_scale: Scale factor for background points
            cur_step: Current training step (used for dropout ratio scheduling)
            
        Returns:
            Tensor of points, potentially concatenated with bkg_points if use_bkg_points is enabled
        """
        points = self.points
        bkg_points_mask = torch.ones_like(points)[:, 0:1]

        if deformed_points is not None:
            assert deformed_points.shape == points.shape
            points = deformed_points
        
        if self.args.geoms.points.use_bkg_points:
            bkg_points = self.bkg_points
            # Add noise to background points if enabled
            if getattr(self.args.geoms.points, 'bkg_points_add_noise', False):
                noise_std = getattr(self.args.geoms.points, 'bkg_points_noise_std', 0.0)
                noise_max_clip = getattr(self.args.geoms.points, 'bkg_points_noise_max_clip', None)
                
                # Generate Gaussian noise
                noise = torch.randn_like(bkg_points) * noise_std
                
                # Clip noise if max_clip is specified
                if noise_max_clip is not None:
                    noise = torch.clamp(noise, -noise_max_clip, noise_max_clip)
                
                bkg_points = bkg_points + noise
            
            points = torch.cat([points, bkg_points], dim=0)
            bkg_points_mask = torch.cat([bkg_points_mask, torch.zeros_like(self.bkg_points)[:, 0:1]], dim=0)
        
        if drop:
            max_ratio = self.get_drop_points_max_ratio(cur_step)
            if max_ratio > 0:
                ratio = random.uniform(0, max_ratio)
                sample_method = getattr(self.args.geoms.points, 'drop_points_sample_method', 'random')

                def _select_keep_indices(points_to_sample, num_keep):
                    if num_keep <= 0:
                        return torch.empty(0, dtype=torch.long, device=points.device)
                    num_points_local = points_to_sample.shape[0]
                    if num_keep >= num_points_local:
                        return torch.arange(num_points_local, device=points.device)
                    if sample_method == 'fps':
                        _, sampled_idx = sample_farthest_points(
                            points=points_to_sample.unsqueeze(0),
                            K=num_keep,
                            random_start_point=getattr(self.args.geoms.points, 'drop_points_fps_random_start', False),
                        )
                        return sampled_idx[0]
                    if sample_method == 'voxel_stratified':
                        voxel_grid_resolution = max(int(getattr(self.args.geoms.points, 'drop_points_voxel_grid_resolution', 64)), 1)
                        voxel_min_keep = max(int(getattr(self.args.geoms.points, 'drop_points_voxel_min_keep', 1)), 1)
                        random_within_voxel = getattr(self.args.geoms.points, 'drop_points_voxel_random_within_voxel', True)
                        voxel_jitter_frac = max(float(getattr(self.args.geoms.points, 'drop_points_voxel_jitter_frac', 0.0)), 0.0)
                        # Non-exact keep-count voxel dropout:
                        # force keep at least voxel_min_keep per occupied voxel,
                        # and apply Bernoulli dropout to all points.
                        dropout_ratio = 1.0 - (float(num_keep) / float(num_points_local))
                        dropout_ratio = min(max(dropout_ratio, 0.0), 1.0)

                        points_min, points_max = torch.aminmax(points_to_sample, dim=0)
                        bbox_extent_max = (points_max - points_min).max()
                        voxel_size = torch.clamp(bbox_extent_max / voxel_grid_resolution, min=1e-8)
                        inv_voxel_size = 1.0 / voxel_size
                        if voxel_jitter_frac > 0:
                            # Jitter voxel origin to reduce persistent grid-aligned artifacts.
                            origin_jitter = (torch.rand(3, device=points.device, dtype=points.dtype) - 0.5) * voxel_size * voxel_jitter_frac
                            points_min = points_min + origin_jitter

                        voxel_coords = torch.floor((points_to_sample - points_min) * inv_voxel_size).to(torch.int64)
                        max_coords = voxel_coords.amax(dim=0) + 1
                        # Collision-free voxel key packing (faster than unique(..., return_inverse=True)).
                        voxel_keys = voxel_coords[:, 0] + voxel_coords[:, 1] * max_coords[0] + voxel_coords[:, 2] * (max_coords[0] * max_coords[1])

                        if random_within_voxel:
                            perm = torch.randperm(num_points_local, device=points.device)
                        else:
                            perm = torch.arange(num_points_local, device=points.device)

                        keys_perm = voxel_keys[perm]
                        sorted_order = torch.argsort(keys_perm)
                        sorted_perm = perm[sorted_order]
                        sorted_keys = keys_perm[sorted_order]

                        random_mask = torch.rand(num_points_local, device=points.device) > dropout_ratio
                        force_keep_mask = torch.zeros(num_points_local, dtype=torch.bool, device=points.device)

                        _, counts = torch.unique_consecutive(sorted_keys, return_counts=True)
                        start_indices = torch.cat([
                            counts.new_zeros(1),
                            torch.cumsum(counts, dim=0)[:-1],
                        ])

                        if voxel_min_keep == 1:
                            force_keep_mask[sorted_perm[start_indices]] = True
                        else:
                            sorted_positions = torch.arange(num_points_local, device=points.device)
                            group_starts_per_point = torch.repeat_interleave(start_indices, counts)
                            local_rank = sorted_positions - group_starts_per_point
                            force_keep_mask[sorted_perm[local_rank < voxel_min_keep]] = True

                        final_mask = random_mask | force_keep_mask
                        return final_mask.nonzero(as_tuple=False).squeeze(1)
                    return torch.randperm(points_to_sample.shape[0], device=points.device)[:num_keep]
                
                if getattr(self.args.geoms.points, 'drop_fg_points_only', False):
                    num_fg = self.points.shape[0]
                    num_keep_fg = int(num_fg * (1 - ratio))
                    if num_keep_fg < num_fg:
                        fg_indices = _select_keep_indices(points[:num_fg], num_keep_fg)
                        if self.args.geoms.points.use_bkg_points:
                            bkg_indices = torch.arange(num_fg, points.shape[0], device=points.device)
                            self.kept_indices = torch.cat([fg_indices, bkg_indices])
                        else:
                            self.kept_indices = fg_indices
                        
                        points = points[self.kept_indices]
                        bkg_points_mask = bkg_points_mask[self.kept_indices]
                    else:
                        self.kept_indices = None
                else:
                    num_points = points.shape[0]
                    num_keep = int(num_points * (1 - ratio))
                    if num_keep < num_points:
                        self.kept_indices = _select_keep_indices(points, num_keep)
                        points = points[self.kept_indices]
                        bkg_points_mask = bkg_points_mask[self.kept_indices]
                    else:
                        self.kept_indices = None
            else:
                self.kept_indices = None
        else:
            self.kept_indices = None

        self.bkg_points_mask = bkg_points_mask
        return points


    def get_pc_feats(self, bkg_color=None):
        """
        Get point cloud features, optionally concatenated with bkg_points_pc_feats.
        
        Returns:
            Tensor of pc_feats, potentially concatenated with bkg_points_pc_feats if use_bkg_points is enabled
        """
        if self.args.geoms.points.use_bkg_points:
            if self.bkg_points_pc_feats_mlp is not None:
                # Use MLP to generate features from background color
                if bkg_color is not None:
                    # bkg_color shape: (3,) or (1, 3) or (N, 3)
                    bkg_color_input = bkg_color.squeeze()
                    if bkg_color_input.dim() == 1:
                        bkg_color_input = bkg_color_input.unsqueeze(0)  # (1, 3)
                else:
                    # Default to white background [1.0, 1.0, 1.0] if no color provided
                    bkg_color_input = torch.ones(1, 3, device=self.device)
                
                # Apply encoding if available
                if self.bkg_points_pc_feats_encoder is not None:
                    encoded_input = self.bkg_points_pc_feats_encoder(bkg_color_input)
                    bkg_pc_feats = self.bkg_points_pc_feats_mlp(encoded_input)  # (1, pc_feat_dim)
                else:
                    bkg_pc_feats = self.bkg_points_pc_feats_mlp(bkg_color_input)  # (1, pc_feat_dim)
            elif self.args.rnd_background and self.args.rnd_background_use_4_feats:
                # Old method: weighted sum of 4 feature vectors
                weights = torch.ones(4, device=self.device)
                if bkg_color is not None:
                    weights[1:] = bkg_color.squeeze()
                bkg_pc_feats = torch.sum(self.bkg_points_pc_feats * weights.unsqueeze(-1), dim=0, keepdim=True)
            else:
                # Old method: single feature vector
                bkg_pc_feats = self.bkg_points_pc_feats
            bkg_pc_feats = bkg_pc_feats.expand(self.bkg_points.shape[0], -1)
            if self.pc_feats.dim() != bkg_pc_feats.dim():
                bkg_pc_feats = bkg_pc_feats.reshape(self.pc_feats.shape)
            pc_feats = torch.cat([self.pc_feats, bkg_pc_feats], dim=0)
        else:
            pc_feats = self.pc_feats
            
        if self.kept_indices is not None:
            pc_feats = pc_feats[self.kept_indices]
            
        return pc_feats


    def get_bkg_sphere_intersection(self, rays_o, rays_d, sphere_center=None, sphere_radius=None):
        """
        Compute the intersection point of rays with the background sphere.
        
        This is a wrapper around the unified get_ray_sphere_intersection function from utils.
        Uses the model's configured sphere parameters by default.
        
        Args:
            rays_o: Ray origins (various shapes supported, see utils.get_ray_sphere_intersection)
            rays_d: Ray directions (will be normalized)
            sphere_center: (3,) center of sphere, or None to use config (default)
            sphere_radius: scalar radius of the sphere, or None to use config (default)
        
        Returns:
            intersection_points: Intersection points with same spatial shape as rays_d
        """
        if sphere_radius is None:
            sphere_radius = self.bkg_sphere_radius * self.coord_scale
        if sphere_center is None:
            sphere_center = [c * self.coord_scale for c in self.bkg_sphere_center]
            
        return get_ray_sphere_intersection(rays_o, rays_d, sphere_center, sphere_radius)


    def sample_cubemap(self, directions, textures=None, activation=None):
        """
        Sample values from cubemap textures given 3D directions.
        
        Uses cube mapping to determine which face each direction points to,
        computes UV coordinates on that face, and samples using bilinear interpolation.
        
        Args:
            directions: Normalized direction vectors of shape (..., 3)
            textures: Optional cubemap textures to sample from (defaults to self.cubemap_textures)
            activation: Optional activation function to apply (defaults to self.cubemap_activation)
            
        Returns:
            torch.Tensor: Sampled values of shape (..., C) where C is the number of channels in textures
        """
        # Use default textures and activation if not provided
        if textures is None:
            textures = self.cubemap_textures
        if activation is None:
            activation = self.cubemap_activation
            
        if textures is None:
            raise ValueError("Cubemap textures are not initialized")
        
        num_channels = textures[0].shape[0]  # Get channel count from texture
            
        original_shape = directions.shape[:-1]
        directions = directions.reshape(-1, 3)  # (N, 3)
        N = directions.shape[0]
        device = directions.device
        
        # Get absolute values for face selection
        abs_dirs = torch.abs(directions)
        
        # Determine dominant axis (which face the direction points to)
        # 0: +X, 1: -X, 2: +Y, 3: -Y, 4: +Z, 5: -Z
        max_axis = torch.argmax(abs_dirs, dim=-1)  # (N,) values in [0, 1, 2] for x, y, z
        
        # Determine sign of dominant axis
        signs = torch.zeros(N, dtype=torch.long, device=device)
        for i in range(3):
            mask = max_axis == i
            signs[mask] = (directions[mask, i] < 0).long()
        
        # Face index: axis * 2 + (1 if negative else 0)
        face_indices = max_axis * 2 + signs  # (N,) values in [0, 5]
        
        # Compute UV coordinates for each face
        x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
        
        # Initialize UV coordinates
        u = torch.zeros(N, device=device)
        v = torch.zeros(N, device=device)
        
        # +X face (index 0): u = -z/x, v = -y/x
        mask = face_indices == 0
        if mask.any():
            ma = torch.abs(x[mask])
            u[mask] = (-z[mask] / ma + 1) / 2
            v[mask] = (-y[mask] / ma + 1) / 2
        
        # -X face (index 1): u = z/(-x), v = -y/(-x)
        mask = face_indices == 1
        if mask.any():
            ma = torch.abs(x[mask])
            u[mask] = (z[mask] / ma + 1) / 2
            v[mask] = (-y[mask] / ma + 1) / 2
        
        # +Y face (index 2): u = x/y, v = z/y
        mask = face_indices == 2
        if mask.any():
            ma = torch.abs(y[mask])
            u[mask] = (x[mask] / ma + 1) / 2
            v[mask] = (z[mask] / ma + 1) / 2
        
        # -Y face (index 3): u = x/(-y), v = -z/(-y)
        mask = face_indices == 3
        if mask.any():
            ma = torch.abs(y[mask])
            u[mask] = (x[mask] / ma + 1) / 2
            v[mask] = (-z[mask] / ma + 1) / 2
        
        # +Z face (index 4): u = x/z, v = -y/z
        mask = face_indices == 4
        if mask.any():
            ma = torch.abs(z[mask])
            u[mask] = (x[mask] / ma + 1) / 2
            v[mask] = (-y[mask] / ma + 1) / 2
        
        # -Z face (index 5): u = -x/(-z), v = -y/(-z)
        mask = face_indices == 5
        if mask.any():
            ma = torch.abs(z[mask])
            u[mask] = (-x[mask] / ma + 1) / 2
            v[mask] = (-y[mask] / ma + 1) / 2
        
        # Clamp UVs to valid range
        u = torch.clamp(u, 0, 1)
        v = torch.clamp(v, 0, 1)
        
        # Sample from each face texture
        output = torch.zeros(N, num_channels, device=device)
        
        for face_idx in range(6):
            mask = face_indices == face_idx
            if not mask.any():
                continue
            
            face_texture = textures[face_idx]  # (C, H, W)
            
            # Get UVs for this face
            face_uvs = torch.stack([u[mask], v[mask]], dim=-1)  # (M, 2)
            
            # Use bilinear interpolation via grid_sample
            face_texture_tensor = face_texture.unsqueeze(0)  # (1, C, H, W)
            
            # Normalize UVs to [-1, 1] for grid_sample
            uvs_normalized = face_uvs * 2 - 1
            uvs_tensor = uvs_normalized.unsqueeze(0).unsqueeze(0)  # (1, 1, M, 2)
            
            sampled = F.grid_sample(face_texture_tensor, uvs_tensor, align_corners=True, 
                                    mode='bilinear', padding_mode='border')
            sampled = sampled.squeeze(0).squeeze(1).permute(1, 0)  # (M, C)
            
            output[mask] = sampled
        
        # Apply activation function if configured
        if activation is not None:
            output = activation(output)
        
        # Reshape back to original shape
        output = output.reshape(*original_shape, num_channels)
        
        return output


    def get_cubemap_background_color(self, rays_o, rays_d, cur_step=-1):
        """
        Get background color from cubemap textures for given rays.
        
        Args:
            rays_o: Ray origins (N, 3) or (N, H, W, 3)
            rays_d: Ray directions (N, H, W, 3)
            cur_step: Current training step
            
        Returns:
            Background colors of shape (N, H, W, 3), or None if cubemap not active
        """
        if self.cubemap_textures is None:
            return None
            
        if cur_step >= 0 and cur_step < self.cubemap_start_step:
            return None
            
        # Get sphere intersection points
        sphere_intersections = self.get_bkg_sphere_intersection(rays_o, rays_d)
        
        # Normalize to get directions from origin
        directions = F.normalize(sphere_intersections, dim=-1)
        
        # Sample colors from cubemap
        cubemap_colors = self.sample_cubemap(directions)
        
        return cubemap_colors


    def get_cubemap_background_features(self, rays_o, rays_d, cur_step=-1):
        """
        Get background features from cubemap feature map for given rays.
        Used to replace append_bkg_points_feats when cubemap_use_feature_map is enabled.
        
        Args:
            rays_o: Ray origins (N, 3) or (N, H, W, 3)
            rays_d: Ray directions (N, H, W, 3)
            cur_step: Current training step
            
        Returns:
            Background features of shape (N, H, W, feature_dim), or None if not active
        """
        if self.cubemap_feature_textures is None or not self.cubemap_use_feature_map:
            return None
            
        if cur_step >= 0 and cur_step < self.cubemap_start_step:
            return None
            
        # Get sphere intersection points
        sphere_intersections = self.get_bkg_sphere_intersection(rays_o, rays_d)
        
        # Normalize to get directions from origin
        directions = F.normalize(sphere_intersections, dim=-1)
        
        # Sample features from cubemap feature textures
        cubemap_features = self.sample_cubemap(directions, 
                                                textures=self.cubemap_feature_textures, 
                                                activation=self.cubemap_feature_activation)
        
        return cubemap_features


    def get_cubemap_depth_offset(self, rays_o, rays_d, cur_step=-1):
        """
        Get depth offset from cubemap depth map for given rays.
        The depth offset represents the distance from the sphere intersection point
        to the actual background surface, along the ray direction.
        
        Args:
            rays_o: Ray origins (N, 3) or (N, H, W, 3)
            rays_d: Ray directions (N, H, W, 3)
            cur_step: Current training step
            
        Returns:
            Depth offset of shape (N, H, W, 1), or None if not active.
            Positive offset means the background is further away than the sphere.
            Negative offset means the background is closer than the sphere.
        """
        if self.cubemap_depth_textures is None or not self.cubemap_use_depth_map:
            return None
            
        if cur_step >= 0 and cur_step < self.cubemap_start_step:
            return None
            
        # Use normalized ray direction for cubemap lookup
        # This makes the depth map view-direction based rather than sphere-intersection based
        directions = F.normalize(rays_d, dim=-1)
        
        # Sample depth offset from cubemap depth textures (no activation - unbounded values)
        # Output shape: (..., 1)
        cubemap_depth = self.sample_cubemap(directions, 
                                            textures=self.cubemap_depth_textures, 
                                            activation=None)
        
        # Scale the depth offset
        # cubemap_depth is unbounded, multiply by scale to control magnitude
        depth_offset = cubemap_depth * self.cubemap_depth_scale * self.coord_scale
        
        return depth_offset


    def get_cubemap_tv_loss(self):
        """
        Compute total variation loss for cubemap textures to encourage smoothness
        while preserving edges. This helps reduce noise in the learned background.
        
        Returns:
            TV loss value, or zero if cubemap textures are not initialized
        """
        tv_loss = torch.zeros(1, device=self.device).sum()
        
        # TV loss for RGB cubemap
        if self.cubemap_textures is not None:
            for texture in self.cubemap_textures:
                # texture shape: (C, H, W)
                # Compute horizontal and vertical differences
                tv_h = torch.abs(texture[:, 1:, :] - texture[:, :-1, :]).mean()
                tv_w = torch.abs(texture[:, :, 1:] - texture[:, :, :-1]).mean()
                tv_loss = tv_loss + tv_h + tv_w
            tv_loss = tv_loss / 6.0  # Average over 6 faces
        
        # TV loss for feature cubemap
        if self.cubemap_feature_textures is not None:
            feature_tv_loss = torch.zeros(1, device=self.device).sum()
            for texture in self.cubemap_feature_textures:
                tv_h = torch.abs(texture[:, 1:, :] - texture[:, :-1, :]).mean()
                tv_w = torch.abs(texture[:, :, 1:] - texture[:, :, :-1]).mean()
                feature_tv_loss = feature_tv_loss + tv_h + tv_w
            tv_loss = tv_loss + feature_tv_loss / 6.0
        
        # TV loss for depth cubemap
        if self.cubemap_depth_textures is not None:
            depth_tv_loss = torch.zeros(1, device=self.device).sum()
            for texture in self.cubemap_depth_textures:
                tv_h = torch.abs(texture[:, 1:, :] - texture[:, :-1, :]).mean()
                tv_w = torch.abs(texture[:, :, 1:] - texture[:, :, :-1]).mean()
                depth_tv_loss = depth_tv_loss + tv_h + tv_w
            tv_loss = tv_loss + depth_tv_loss / 6.0
        
        return tv_loss


    def get_influ_scores(self):
        """
        Get influence scores, optionally concatenated with bkg_points_influ_scores.
        
        Returns:
            Tensor of points_influ_scores, potentially concatenated with bkg_points_influ_scores if use_bkg_points is enabled
        """
        if self.args.geoms.points.use_bkg_points:
            bkg_points_influ_scores = self.bkg_points_influ_scores.expand(self.bkg_points.shape[0], -1)
            cur_points_influ_score = torch.cat([self.points_influ_scores, bkg_points_influ_scores], dim=-2)
        else:
            cur_points_influ_score = self.points_influ_scores
            
        if self.kept_indices is not None:
            cur_points_influ_score = cur_points_influ_score[self.kept_indices]

        return cur_points_influ_score

    def get_scaler(self):
        """
        Get point scaler, optionally concatenated with bkg_points_scaler if use_bkg_points is enabled.
        
        Returns:
            Tensor of points_scaler, potentially concatenated with bkg_points_scaler if use_bkg_points is enabled
        """
        if self.args.geoms.points.use_bkg_points:
            # For background points, use scaler of 1.0 (no scaling)
            bkg_points_scaler = torch.ones(self.bkg_points.shape[0], 1, device=self.points_scaler.device)
            cur_points_scaler = torch.cat([self.points_scaler, bkg_points_scaler], dim=-2)
        else:
            cur_points_scaler = self.points_scaler
            
        if self.kept_indices is not None:
            cur_points_scaler = cur_points_scaler[self.kept_indices]

        return cur_points_scaler


    @torch.no_grad()
    def _get_topk_point_inds(self, rays_o, rays_d, points, pix_coords, points_2d, z, step=-1, surface_points=None, normals=None, rays_d_no_norm=None, points_sampled=None):
        """
        Select the top-k points with the smallest distance to the rays from all points
        """
        N, H, W, _ = rays_d.shape
        num_pts, _ = points.shape
        if rays_o.dim() == 2:
            rays_o_base = rays_o
            rays_o_hw = rays_o.reshape(N, 1, 1, 3).expand(-1, H, W, -1)
        elif rays_o.dim() == 4:
            rays_o_hw = rays_o
            rays_o_base = rays_o[:, 0, 0, :]
        else:
            raise ValueError(f"Unsupported rays_o ndim={rays_o.dim()}, expected 2 or 4")
        rays_o_expanded = rays_o_hw.unsqueeze(-2)  # (N, H, W, 1, 3)

        if z is not None and z.ndim == 2:
            z = z[:, None, None, :]

        min_d2r = torch.zeros(N, H, W, device=self.device)
        if self.select_k >= num_pts or self.select_k < 0:
            select_k_ind = torch.arange(num_pts, device=self.device).expand(N, H, W, -1)
            min_d2r = torch.zeros(N, H, W, device=self.device)
        # elif surface_points is not None:
        #     # use KNN to get the top-k points
        #     _, select_k_ind, _ = knn_points(surface_points.reshape(1, N*H*W, 3), points.reshape(1, num_pts, 3), K=self.select_k, return_sorted=False)
        #     min_d2r = torch.zeros(N, H, W, device=self.device)
        #     select_k_ind = select_k_ind.reshape(N, H, W, self.select_k)
        elif self.args.geoms.points.select_k_type == "knn_frustum":
            # Filter points inside the camera frustum and use KNN (vectorized with lengths2)
            # z is the z-coordinate in camera space (negative = in front of camera in Blender convention)
            # points_2d contains projected 2D coordinates (u, v) on image plane
            
            # Benchmark: frustum mask computation
            if self._benchmark_knn_frustum:
                torch.cuda.synchronize()
                _t_frustum_start = time.perf_counter()
                
            # Get 2D coordinates: handle (N, num_pts, 2) or (num_pts, 2)
            if points_2d.ndim == 3:
                pts_2d = points_2d  # (N, num_pts, 2)
            else:
                pts_2d = points_2d.unsqueeze(0).expand(N, -1, -1)  # (N, num_pts, 2)
            
            # Get z values: z has shape (N, 1, 1, num_pts)
            z_vals = z[:, 0, 0, :]  # (N, num_pts)
                            
            # Create frustum mask (vectorized)
            # Check 1: Point must be in front of camera (z < 0 in Blender convention)
            in_front = z_vals < 0 - self.eps  # (N, num_pts)
            # Check 2: u must be within [0, W]
            in_width = (pts_2d[..., 0] >= 0 - self.pixel_frustum_margin) & (pts_2d[..., 0] < self.W + self.pixel_frustum_margin)  # (N, num_pts)
            # Check 3: v must be within [0, H]
            in_height = (pts_2d[..., 1] >= 0 - self.pixel_frustum_margin) & (pts_2d[..., 1] < self.H + self.pixel_frustum_margin)  # (N, num_pts)
            
            frustum_mask = in_front & in_width & in_height  # (N, num_pts)
                            
            # Count valid points per batch element
            lengths2 = frustum_mask.sum(dim=-1)  # (N,)
            max_valid = lengths2.max().item()
            
            if self._benchmark_knn_frustum:
                torch.cuda.synchronize()
                _t_frustum_end = time.perf_counter()
                _t_frustum = (_t_frustum_end - _t_frustum_start) * 1000  # ms
            
            # Get depth scaling mode
            depth_scale_mode = getattr(self.args.geoms.points, 'knn_frustum_depth_scale', 'none')
            
            # Helper function to compute depth-scaled d2r
            def compute_scaled_d2r(pts_2d_in, pix_coords_in, z_vals_in, depth_scale_mode_in):
                """Compute UV distance optionally scaled by depth."""
                if pts_2d_in.ndim == 3:
                    d2r = torch.norm(pts_2d_in.to(getattr(torch, self.args.topk_dtype))[:, None, None, :, :] - pix_coords_in.to(getattr(torch, self.args.topk_dtype))[..., None, :], dim=-1)  # (N, H, W, num_pts)
                else:
                    d2r = torch.norm(pts_2d_in.to(getattr(torch, self.args.topk_dtype)) - pix_coords_in.to(getattr(torch, self.args.topk_dtype))[..., None, :], dim=-1)  # (N, H, W, num_pts)
                
                if depth_scale_mode_in == "none":
                    return d2r
                
                # Get depth values: z_vals has shape (N, num_pts), need to expand to (N, H, W, num_pts)
                # z is negative for points in front of camera (Blender convention), so we use -z as depth
                depth = (-z_vals_in)[:, None, None, :]  # (N, 1, 1, num_pts)
                depth = depth.expand(-1, H, W, -1)  # (N, H, W, num_pts)
                
                if depth_scale_mode_in == "approximate":
                    # Approximate: d2r_scaled = d2r * depth
                    # This approximates 3D point-to-ray distance (ignoring focal length normalization since it's constant)
                    d2r_scaled = d2r * depth.to(d2r.dtype)
                elif depth_scale_mode_in == "exact":
                    # Exact: d2r_scaled = d2r * depth * ray_length_factor
                    # ray_length_factor accounts for off-axis rays
                    # ray_length_factor = sqrt(1 + ((px - cx)/fx)^2 + ((py - cy)/fy)^2)
                    px = pix_coords_in[..., 0:1]  # (H, W, 1)
                    py = pix_coords_in[..., 1:2]  # (H, W, 1)
                    sx = (px - self.cx) / self.fx  # (H, W, 1)
                    sy = (py - self.cy) / self.fy  # (H, W, 1)
                    ray_length_factor = torch.sqrt(1.0 + sx**2 + sy**2)  # (H, W, 1)
                    ray_length_factor = ray_length_factor.unsqueeze(0).to(d2r.dtype)  # (1, H, W, 1)
                    d2r_scaled = d2r * depth.to(d2r.dtype) * ray_length_factor
                else:
                    raise ValueError(f"Unknown knn_frustum_depth_scale mode: {depth_scale_mode_in}")
                
                return d2r_scaled
            
            if max_valid == 0:
                # Fallback: if no points in frustum for any batch, use d2r method
                d2r = compute_scaled_d2r(pts_2d, pix_coords, z_vals, depth_scale_mode)
                _, select_k_ind = torch.topk(d2r, self.select_k, dim=-1, largest=False, sorted=False)
            elif max_valid < self.select_k:
                # If all batches have fewer than select_k points, use d2r with frustum masking
                d2r = compute_scaled_d2r(pts_2d, pix_coords, z_vals, depth_scale_mode)
                d2r_masked = d2r.clone()
                frustum_mask_expanded = frustum_mask[:, None, None, :].expand_as(d2r)
                d2r_masked[~frustum_mask_expanded] = 1e10
                _, select_k_ind = torch.topk(d2r_masked, self.select_k, dim=-1, largest=False, sorted=False)
            elif depth_scale_mode != "none":
                # Depth scaling is enabled - use explicit distance computation with topk
                # (knn_points cannot handle depth-weighted distances directly)
                if self._benchmark_knn_frustum:
                    torch.cuda.synchronize()
                    _t_depth_scale_start = time.perf_counter()
                
                d2r = compute_scaled_d2r(pts_2d, pix_coords, z_vals, depth_scale_mode)
                d2r_masked = d2r.clone()
                frustum_mask_expanded = frustum_mask[:, None, None, :].expand_as(d2r)
                d2r_masked[~frustum_mask_expanded] = 1e10
                _, select_k_ind = torch.topk(d2r_masked, self.select_k, dim=-1, largest=False, sorted=False)
                
                if self._benchmark_knn_frustum:
                    torch.cuda.synchronize()
                    _t_depth_scale_end = time.perf_counter()
                    _t_depth_scale = (_t_depth_scale_end - _t_depth_scale_start) * 1000  # ms
                    print(f"[knn_frustum] frustum_mask: {_t_frustum:.3f}ms | depth_scale_topk ({depth_scale_mode}): {_t_depth_scale:.3f}ms | total: {_t_frustum + _t_depth_scale:.3f}ms | max_valid: {max_valid}/{num_pts}")
            else:
                # Benchmark: sort operation
                if self._benchmark_knn_frustum:
                    torch.cuda.synchronize()
                    _t_sort_start = time.perf_counter()
                
                # Sort frustum_mask so valid points come first (descending sort of mask)
                # This gives us indices that reorder points: valid first, invalid last
                sorted_indices = torch.argsort(frustum_mask.float(), dim=-1, descending=True)  # (N, num_pts)
                
                # Gather pts_2d in sorted order: valid points first
                sorted_pts_2d = torch.gather(pts_2d, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, 2))  # (N, num_pts, 2)
                
                # Truncate to max_valid points for efficiency
                sorted_pts_2d = sorted_pts_2d[:, :max_valid, :]  # (N, max_valid, 2)
                
                # Clamp lengths2 to max_valid (should already be <= max_valid, but for safety)
                lengths2_clamped = lengths2.clamp(min=1, max=max_valid)  # (N,) - clamp min to 1 to avoid edge cases
                
                # Query points: pixel coordinates flattened
                query_pts = pix_coords.reshape(N, H * W, 2)  # (N, H*W, 2)
                
                if self._benchmark_knn_frustum:
                    torch.cuda.synchronize()
                    _t_sort_end = time.perf_counter()
                    _t_sort = (_t_sort_end - _t_sort_start) * 1000  # ms
                
                # Benchmark: KNN operation
                if self._benchmark_knn_frustum:
                    torch.cuda.synchronize()
                    _t_knn_start = time.perf_counter()
                                    
                # Use KNN with lengths2 to handle variable valid counts per batch
                _, knn_idx, _ = knn_points(
                    query_pts.to(getattr(torch, self.args.topk_dtype)),
                    sorted_pts_2d.to(getattr(torch, self.args.topk_dtype)),
                    lengths2=lengths2_clamped,
                    K=self.select_k,
                    return_sorted=False
                )
                # knn_idx: (N, H*W, K) - indices into sorted_pts_2d
                
                if self._benchmark_knn_frustum:
                    torch.cuda.synchronize()
                    _t_knn_end = time.perf_counter()
                    _t_knn = (_t_knn_end - _t_knn_start) * 1000  # ms
                
                # Benchmark: index mapping operation
                if self._benchmark_knn_frustum:
                    torch.cuda.synchronize()
                    _t_gather_start = time.perf_counter()
                
                # Map KNN indices back to original point indices
                # knn_idx indexes into sorted order, sorted_indices maps sorted -> original
                # We need to gather from sorted_indices using knn_idx
                sorted_indices_truncated = sorted_indices[:, :max_valid]  # (N, max_valid)
                                    
                # Expand for gathering: (N, H*W, K) indices into (N, max_valid) -> need (N, H*W*K)
                knn_idx_flat = knn_idx.reshape(N, -1)  # (N, H*W*K)
                select_k_ind_flat = torch.gather(sorted_indices_truncated, 1, knn_idx_flat)  # (N, H*W*K)
                select_k_ind = select_k_ind_flat.reshape(N, H, W, self.select_k)
                
                if self._benchmark_knn_frustum:
                    torch.cuda.synchronize()
                    _t_gather_end = time.perf_counter()
                    _t_gather = (_t_gather_end - _t_gather_start) * 1000  # ms
                    
                    # Print benchmark results
                    print(f"[knn_frustum] frustum_mask: {_t_frustum:.3f}ms | sort+gather: {_t_sort:.3f}ms | knn: {_t_knn:.3f}ms | index_map: {_t_gather:.3f}ms | total: {_t_frustum + _t_sort + _t_knn + _t_gather:.3f}ms | max_valid: {max_valid}/{num_pts}")
        elif (
            not self.args.geoms.points.project
            and self.args.geoms.points.select_k_type == "d2r_filter"
        ):
            select_k_ind, min_d2r = _select_d2r_filter_chunked(
                rays_o_hw,
                rays_d,
                points,
                self.select_k,
                self.eps,
            )
        elif self.args.geoms.points.project:
            if points_2d.ndim == 3:
                d2r = torch.norm(points_2d.to(getattr(torch, self.args.topk_dtype))[:, None, None, :, :] - pix_coords.to(getattr(torch, self.args.topk_dtype))[..., None, :], dim=-1)  # (N, H, W, num_pts)
            else:
                d2r = torch.norm(points_2d.to(getattr(torch, self.args.topk_dtype)) - pix_coords.to(getattr(torch, self.args.topk_dtype))[..., None, :], dim=-1)  # (N, H, W, num_pts)
            min_d2r = d2r.min(dim=-1)[0]
            # The historical "nn" selector required RAPIDS cuml/cupy, which the
            # release does not depend on and no released config selects.
            if self.args.geoms.points.select_k_type == "d2r":
                _, select_k_ind = torch.topk(d2r, self.select_k, dim=-1, largest=False, sorted=False)
            elif self.args.geoms.points.select_k_type == "d2r_z":
                # print("z", z.shape, "d2r", d2r.shape)
                feature = d2r * (1 - z).pow(self.args.geoms.points.select_k_z_pow)
                # feature[(z > -1).expand_as(feature)] = 1e10
                # print("z", z.min(), z.max())
                _, select_k_ind = torch.topk(feature, self.select_k, dim=-1, largest=False, sorted=False)
            elif self.args.geoms.points.select_k_type == "d2r_z_filter":
                # print("z", z.shape, "d2r", d2r.shape)
                feature = d2r * (1 - z).pow(self.args.geoms.points.select_k_z_pow)
                feature[(z > -1).expand_as(feature)] = 1e10
                # print("z", z.min(), z.max())
                _, select_k_ind = torch.topk(feature, self.select_k, dim=-1, largest=False, sorted=False)
            elif self.args.geoms.points.select_k_type == "d2r_z_filter2":
                # print("z", z.shape, "d2r", d2r.shape)
                feature = d2r * (1 - z).pow(self.args.geoms.points.select_k_z_pow)
                # feature[(z > -1).expand_as(feature)] = 1e10
                camera_to_points = points.reshape(1, 1, 1, -1, 3) - rays_o_expanded
                cosine_phi = torch.sum(normalize_vector(camera_to_points) * normalize_vector(rays_d).reshape(N, H, W, 1, 3), dim=-1)
                feature[cosine_phi <= 0] = 1e10
                # print("z", z.min(), z.max())
                _, select_k_ind = torch.topk(feature, self.select_k, dim=-1, largest=False, sorted=False)                                    
            elif self.args.geoms.points.select_k_type == "thresh_pd":
                pd = torch.norm(points.reshape(1, 1, 1, -1, 3) - rays_o_expanded, dim=-1).repeat(1, H, W, 1)
                count = (d2r < self.args.geoms.points.d2r_thresh).sum(dim=-1)
                # print("count", count.shape, count.min(), count.max())
                # exit(0)
                pd[d2r > self.args.geoms.points.d2r_thresh] = pd[d2r > self.args.geoms.points.d2r_thresh] * 1000
                _, select_k_ind = torch.topk(pd, self.select_k, dim=-1, largest=False, sorted=False)
            elif self.args.geoms.points.select_k_type == "nn1":
                pd = torch.norm(points.reshape(1, -1, 3) - rays_o_base.reshape(N, 1, 3), dim=-1)
                near = pd.min(dim=-1)[0].detach().cpu().numpy()
                far = pd.max(dim=-1)[0].detach().cpu().numpy()
                pd_sampled = np.linspace(near, far, num=self.select_k, endpoint=True, axis=-1)
                pd_sampled = torch.from_numpy(pd_sampled).to(self.device)
                if points_sampled is None:
                    points_sampled = rays_o_expanded + pd_sampled.reshape(N, 1, 1, self.select_k, 1) * rays_d.reshape(N, H, W, 1, 3)
                self.points_sampled = points_sampled
                _, select_k_ind, _ = knn_points(points_sampled.reshape(1, N*H*W*self.select_k, 3), points.reshape(1, num_pts, 3), K=1, return_sorted=False)
                select_k_ind = select_k_ind.reshape(N, H, W, self.select_k)
            elif self.args.geoms.points.select_k_type == "nn2":
                # TODO: 
                feature = d2r * (1 - z).pow(self.args.geoms.points.select_k_z_pow)
                _, select_k_ind = torch.topk(feature, self.select_k, dim=-1, largest=False, sorted=False)
                pd = torch.norm(points.reshape(1, -1, 3) - rays_o_base.reshape(N, 1, 3), dim=-1)
                near = pd.min(dim=-1)[0].detach().cpu().numpy()
                far = pd.max(dim=-1)[0].detach().cpu().numpy()
                pd_sampled = np.linspace(near, far, num=self.select_k, endpoint=True, axis=-1)
                pd_sampled = torch.from_numpy(pd_sampled).to(self.device)
                if points_sampled is None:
                    points_sampled = rays_o_expanded + pd_sampled.reshape(N, 1, 1, self.select_k, 1) * rays_d.reshape(N, H, W, 1, 3)
                self.points_sampled = points_sampled
                _, select_k_ind, _ = knn_points(points_sampled.reshape(1, N*H*W*self.select_k, 3), points.reshape(1, num_pts, 3), K=1, return_sorted=False)
                select_k_ind = select_k_ind.reshape(N, H, W, self.select_k)
            # elif self.args.geoms.points.select_k_type == "d2r_z_if":
            #     # print("z", z.shape, "d2r", d2r.shape)
            #     feature = d2r * (1 - z).pow(self.args.geoms.points.select_k_z_pow)
            #     feature = feature * (self.points_influ_scores.max() - self.points_influ_scores.reshape(1, 1, 1, -1) + 1e-4)     
            #     # feature[(z > -1).expand_as(feature)] = 1e10
            #     # print("z", z.min(), z.max())
            #     _, select_k_ind = torch.topk(feature, self.select_k, dim=-1, largest=False, sorted=False)
            # elif self.args.geoms.points.select_k_type == "d2r_z_if_before_prune":
            #     # print("z", z.shape, "d2r", d2r.shape)
            #     feature = d2r * (1 - z).pow(self.args.geoms.points.select_k_z_pow)
            #     if step > 0 and step < self.args.training.prune_start and not self.pruned_points:
            #         feature = feature * (self.points_influ_scores.max() - self.points_influ_scores.reshape(1, 1, 1, -1) + 1e-4)
            #     # feature = feature * (self.points_influ_scores.max() - self.points_influ_scores.reshape(1, 1, 1, -1) + 1e-4)
            #     # feature[(z > -1).expand_as(feature)] = 1e10
            #     # print("z", z.min(), z.max())
            #     _, select_k_ind = torch.topk(feature, self.select_k, dim=-1, largest=False, sorted=False)
            elif self.args.geoms.points.select_k_type == "d2r_z2":
                _, select_k_ind = torch.topk(d2r / (1 - z).pow(self.args.geoms.points.select_k_z_pow), self.select_k, dim=-1, largest=False, sorted=False)
            elif self.args.geoms.points.select_k_type == "d2r_cone":
                z_max = (1 - z).max()
                d2r = d2r / (z_max - (1 - z) + self.eps).pow(self.args.geoms.points.select_k_cone_pow)
                _, select_k_ind = torch.topk(d2r, self.select_k, dim=-1, largest=False, sorted=False)
            elif self.args.geoms.points.select_k_type == "d2r_z_cone":
                z_max = (1 - z).max()
                d2r = d2r / (z_max - (1 - z) + self.eps).pow(self.args.geoms.points.select_k_cone_pow)
                _, select_k_ind = torch.topk(d2r * (1 - z).pow(self.args.geoms.points.select_k_z_pow), self.select_k, dim=-1, largest=False, sorted=False)
            elif self.args.geoms.points.select_k_type == "mlp1":    # 2 features
                features = self.topk_mlp.get_features([d2r, (1 - z).pow(self.args.geoms.points.select_k_z_pow)])
                _, select_k_ind = torch.topk(features, self.select_k, dim=-1, largest=False, sorted=False)
            elif self.args.geoms.points.select_k_type == "mlp2":    # 2 features
                max_z = (1 - z).max()
                features = self.topk_mlp.get_features([d2r, (max_z - 1 + z + 1e-8).pow(self.args.geoms.points.select_k_cone_pow)])
                _, select_k_ind = torch.topk(features, self.select_k, dim=-1, largest=False, sorted=False)
            elif self.args.geoms.points.select_k_type == "mlp3":    # 2 features
                features = self.topk_mlp.get_features([d2r / (1 - z).pow(self.args.geoms.points.select_k_z_pow), 
                                                       (1 - z).pow(self.args.geoms.points.select_k_z_pow)])
                _, select_k_ind = torch.topk(features, self.select_k, dim=-1, largest=False, sorted=False)
            elif self.args.geoms.points.select_k_type == "mlp4":    # 3 features
                d2o_2d = torch.norm(torch.cat([points_2d[..., :2] / z, torch.ones_like(z)[..., None]], dim=-1), dim=-1)
                features = self.topk_mlp.get_features([d2r / (1 - z).pow(self.args.geoms.points.select_k_z_pow), d2o_2d,
                                                       (1 - z).pow(self.args.geoms.points.select_k_z_pow)])
                _, select_k_ind = torch.topk(features, self.select_k, dim=-1, largest=False, sorted=False)
            elif self.args.geoms.points.select_k_type == "mlp5":    # 3 features
                d2o_3d = torch.norm(points - rays_o_base, dim=-1)
                features = self.topk_mlp.get_features([d2r / (1 - z).pow(self.args.geoms.points.select_k_z_pow), d2o_3d,
                                                       (1 - z).pow(self.args.geoms.points.select_k_z_pow)])
                _, select_k_ind = torch.topk(features, self.select_k, dim=-1, largest=False, sorted=False)
            elif self.args.geoms.points.select_k_type == "mlp6":    # 4 features
                d2o_2d = torch.norm(torch.cat([points_2d / z, torch.ones_like(z)[..., None]], dim=-1), dim=-1)
                d2o_3d = torch.norm(points - rays_o_base, dim=-1)
                features = self.topk_mlp.get_features([d2r / (1 - z).pow(self.args.geoms.points.select_k_z_pow), 
                                                       d2o_2d, d2o_3d,
                                                       (1 - z).pow(self.args.geoms.points.select_k_z_pow)])
                _, select_k_ind = torch.topk(features, self.select_k, dim=-1, largest=False, sorted=False)
            elif self.args.geoms.points.select_k_type == "test":
                # features = d2r * 2.003 + pd * -0.00949
                features = d2r * 0.625 + pd * 0.00162
                _, select_k_ind = torch.topk(features, self.select_k, dim=-1, largest=False, sorted=False)
        else:
            pd, d2r, _, _ = self.proximity_attn.get_features(rays_o, rays_d, points)
            pd = pd.squeeze(-1)
            d2r = d2r.squeeze(-1)
            min_d2r = d2r.min(dim=-1)[0]
            if self.args.geoms.points.select_k_type == "d2r":
                features = d2r
                # cos_phi = torch.sum(normalize_vector(-rays_o, self.eps) * normalize_vector(points - rays_o, self.eps), dim=-1).expand_as(features)
                # features[cos_phi <= 0] = 1e10
                _, select_k_ind = torch.topk(features, self.select_k, dim=-1, largest=False, sorted=False)
            elif self.args.geoms.points.select_k_type == "nn1":
                near = pd.min(dim=-1)[0].detach().cpu().numpy()
                far = pd.max(dim=-1)[0].detach().cpu().numpy()
                pd_sampled = np.linspace(near, far, num=self.select_k, endpoint=True, axis=-1)
                pd_sampled = torch.from_numpy(pd_sampled).to(self.device)
                if points_sampled is None:
                    points_sampled = rays_o_expanded + pd_sampled.reshape(N, H, W, self.select_k, 1) * rays_d.reshape(N, H, W, 1, 3)
                self.points_sampled = points_sampled
                _, select_k_ind, _ = knn_points(points_sampled.reshape(1, N*H*W*self.select_k, 3), points.reshape(1, num_pts, 3), K=1, return_sorted=False)
                select_k_ind = select_k_ind.reshape(N, H, W, self.select_k)
            elif self.args.geoms.points.select_k_type == "nn3":
                near_z = (-z).min(dim=-1)[0].detach().cpu().numpy()
                far_z = (-z).max(dim=-1)[0].detach().cpu().numpy()
                z_sampled = np.linspace(near_z, far_z, num=self.select_k, endpoint=True, axis=-1)
                z_sampled = torch.from_numpy(z_sampled).to(self.device)
                if points_sampled is None:
                    points_sampled = rays_o_expanded + z_sampled.reshape(N, 1, 1, self.select_k, 1) * rays_d_no_norm.reshape(N, H, W, 1, 3)
                self.points_sampled = points_sampled
                _, select_k_ind, _ = knn_points(points_sampled.reshape(1, N*H*W*self.select_k, 3), points.reshape(1, num_pts, 3), K=1, return_sorted=False)
                select_k_ind = select_k_ind.reshape(N, H, W, self.select_k)
            elif self.args.geoms.points.select_k_type == "nn4":
                feature = d2r * (1 - z).pow(self.args.geoms.points.select_k_z_pow)
                _, select_k_ind = torch.topk(feature, self.select_k, dim=-1, largest=False, sorted=False)
                # selected_z = z[select_k_ind]
                selected_z = torch.gather(z.expand(N, H, W, -1), dim=-1, index=select_k_ind)
                near_z = (-selected_z).min(dim=-1)[0].detach().cpu().numpy()
                far_z = (-selected_z).max(dim=-1)[0].detach().cpu().numpy()
                z_sampled = np.linspace(near_z, far_z, num=self.select_k, endpoint=True, axis=-1)
                z_sampled = torch.from_numpy(z_sampled).to(self.device)
                if points_sampled is None:
                    points_sampled = rays_o_expanded + z_sampled.reshape(N, H, W, self.select_k, 1) * rays_d_no_norm.reshape(N, H, W, 1, 3)                
                self.points_sampled = points_sampled
                _, select_k_ind, _ = knn_points(points_sampled.reshape(1, N*H*W*self.select_k, 3), points.reshape(1, num_pts, 3), K=1, return_sorted=False)
                select_k_ind = select_k_ind.reshape(N, H, W, self.select_k)
            elif self.args.geoms.points.select_k_type == "nn2":
                feature = d2r * (1 - z).pow(self.args.geoms.points.select_k_z_pow)
                _, select_k_ind = torch.topk(feature, self.select_k, dim=-1, largest=False, sorted=False)
                near = pd.min(dim=-1)[0].detach().cpu().numpy()
                far = pd.max(dim=-1)[0].detach().cpu().numpy()
                pd_sampled = np.linspace(near, far, num=self.select_k, endpoint=True, axis=-1)
                pd_sampled = torch.from_numpy(pd_sampled).to(self.device)
                points_sampled = rays_o_expanded + pd_sampled.reshape(N, 1, 1, self.select_k, 1) * rays_d.reshape(N, H, W, 1, 3)
                _, select_k_ind, _ = knn_points(points_sampled.reshape(1, N*H*W*self.select_k, 3), points.reshape(1, num_pts, 3), K=1, return_sorted=False)
                select_k_ind = select_k_ind.reshape(N, H, W, self.select_k)
            elif self.args.geoms.points.select_k_type == "d2r_z":
                _, select_k_ind = torch.topk(d2r * (1 - z).pow(self.args.geoms.points.select_k_z_pow), self.select_k, dim=-1, largest=False, sorted=False)
            elif self.args.geoms.points.select_k_type == "d2r_cone":
                max_pd = torch.max(pd, dim=-1, keepdim=True)[0]
                d2r = d2r / (max_pd - pd + 1e-8).pow(self.args.geoms.points.select_k_cone_pow)
                d2r[d2r < 0] = 1e10
                _, select_k_ind = torch.topk(d2r, self.select_k, dim=-1, largest=False, sorted=False)
            elif self.args.geoms.points.select_k_type == "d2r_z_cone":
                max_pd = torch.max(pd, dim=-1, keepdim=True)[0]
                d2r = d2r / (max_pd - pd + 1e-8).pow(self.args.geoms.points.select_k_cone_pow)
                d2r[d2r < 0] = 1e6
                _, select_k_ind = torch.topk(d2r * (1 - z).pow(self.args.geoms.points.select_k_z_pow), self.select_k, dim=-1, largest=False, sorted=False)
            elif self.args.geoms.points.select_k_type == "mlp1":    # 2 features
                features = self.topk_mlp.get_features([d2r, pd])
                # cos_phi = torch.sum(normalize_vector(-rays_o, self.eps) * normalize_vector(points - rays_o, self.eps), dim=-1).expand_as(features)
                # features[cos_phi <= 0] = 1e10
                # print("cos_phi", cos_phi.min(), cos_phi.max())
                _, select_k_ind = torch.topk(features, self.select_k, dim=-1, largest=False, sorted=False)
            elif self.args.geoms.points.select_k_type == "mlp2":    # 2 features
                max_pd = torch.max(pd, dim=-1, keepdim=True)[0]
                features = self.topk_mlp.get_features([d2r, (max_pd - pd + 1e-8).pow(self.args.geoms.points.select_k_cone_pow)])
                _, select_k_ind = torch.topk(features, self.select_k, dim=-1, largest=False, sorted=False)
            elif self.args.geoms.points.select_k_type == "mlp3":    # 2 features
                min_pd = torch.min(pd, dim=-1, keepdim=True)[0]
                features = self.topk_mlp.get_features([d2r, (pd - min_pd + 1).pow(self.args.geoms.points.select_k_cone_pow)])
                _, select_k_ind = torch.topk(features, self.select_k, dim=-1, largest=False, sorted=False)
            elif self.args.geoms.points.select_k_type == "mlp4":    # 3 features
                features = self.topk_mlp.get_features([d2r, pd, 1 / (pd + self.eps)])
                _, select_k_ind = torch.topk(features, self.select_k, dim=-1, largest=False, sorted=False)
            elif self.args.geoms.points.select_k_type == "mlp5":    # 5 features
                max_pd = torch.max(pd, dim=-1, keepdim=True)[0]
                features = self.topk_mlp.get_features([d2r, pd, max_pd - pd, 1 / (pd + self.eps), 1 / (max_pd - pd + self.eps)])
                _, select_k_ind = torch.topk(features, self.select_k, dim=-1, largest=False, sorted=False)
            elif self.args.geoms.points.select_k_type == "mlp6":    # 3 features
                max_pd = torch.max(pd, dim=-1, keepdim=True)[0]
                features = self.topk_mlp.get_features([d2r, pd, 1 / (max_pd - pd + self.eps)])
                _, select_k_ind = torch.topk(features, self.select_k, dim=-1, largest=False, sorted=False)
            elif self.args.geoms.points.select_k_type == "mlp7":    # 7 features
                max_pd = torch.max(pd, dim=-1, keepdim=True)[0]
                features = self.topk_mlp.get_features([d2r, pd, max_pd - pd, 1 / (pd + self.eps), 1 / (max_pd - pd + self.eps), 1 / (pd + self.eps) ** 2, 1 / (max_pd - pd + self.eps) ** 2])
                _, select_k_ind = torch.topk(features, self.select_k, dim=-1, largest=False, sorted=False)
            elif self.args.geoms.points.select_k_type == "mlp8":    # 6 features
                max_pd = torch.max(pd, dim=-1, keepdim=True)[0]
                features = self.topk_mlp.get_features([d2r, pd, max_pd - pd, d2r ** 2, pd ** 2, (max_pd - pd) ** 2])
                _, select_k_ind = torch.topk(features, self.select_k, dim=-1, largest=False, sorted=False)
            elif self.args.geoms.points.select_k_type == "test":
                # features = d2r * 2.003 + pd * -0.00949
                features = d2r * 0.625 + pd * 0.00162
                _, select_k_ind = torch.topk(features, self.select_k, dim=-1, largest=False, sorted=False)
            else:
                raise ValueError(f"Unknown select_k_type: {self.args.geoms.points.select_k_type}")
        self.select_k_ind = select_k_ind
        return select_k_ind, min_d2r
    

    @property
    def get_points_alpha(self):
        if self.points_alpha is None:
            return None
        return self.points_alpha_act(self.points_alpha)


    def add_gaussian_noise_to_points(self, points, step):
        noise_std = self.args.geoms.points.noise_std
        # Apply annealing to noise_std if specified
        if self.args.geoms.points.noise_annealing_type is not None and self.args.training.steps > 0:
            start_noise_std = self.args.geoms.points.noise_std
            end_noise_std = self.args.geoms.points.end_noise_std
            
            progress = min(step / self.args.training.steps, 1.0)  # Clamp to [0, 1]
            
            if self.args.geoms.points.noise_annealing_type == 'cosine':
                # Cosine annealing: smooth decay from start to end
                noise_std = end_noise_std + (start_noise_std - end_noise_std) * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
            elif self.args.geoms.points.noise_annealing_type == 'linear':
                # Linear annealing: linear decay from start to end
                noise_std = start_noise_std + (end_noise_std - start_noise_std) * progress
            else:
                raise ValueError(f"Unknown annealing type: {self.args.geoms.points.noise_annealing_type}. Use 'cosine' or 'linear'.")
        
        # Generate 3D Gaussian noise with zero mean and (possibly annealed) std
        if step % 200 == 0:
            print("Adding noise to points, noise_std:", noise_std.item())
        noise = torch.randn_like(points) * noise_std
        
        # Truncate noise by norm if max_noise_norm is specified
        if self.args.geoms.points.max_noise_norm is not None:
            noise_norms = torch.norm(noise, dim=-1, keepdim=True)
            # Scale down noise vectors that exceed the maximum norm
            scale_factor = torch.clamp(self.args.geoms.points.max_noise_norm / (noise_norms + 1e-8), max=1.0)
            noise = noise * scale_factor
        
        # Add noise to points
        noisy_points = points + noise
        
        return noisy_points


    def reset_alphas(self):
        if self.points_alpha is not None:
            alphas_new = self.inverse_alpha_activation(torch.min(self.get_points_alpha, torch.ones_like(self.get_points_alpha) * self.args.geoms.alpha.reset_val))
            optimizable_tensors = self.replace_tensor_to_optimizer(alphas_new, "points_alpha")
            self.points_alpha = optimizable_tensors["points_alpha"]
            print("Reset alphas to", self.points_alpha.shape, self.points_alpha.min().item(), self.points_alpha.max().item(), self.points_alpha.mean().item(), self.points_alpha.std().item())
            
    
    # def replace_tensor_to_optimizer(self, tensor, name):
    #     optimizable_tensors = {}
    #     for group in self.optimizer.param_groups:
    #         if group["name"] == name:
    #             stored_state = self.optimizer.state.get(group['params'][0], None)
    #             stored_state["exp_avg"] = torch.zeros_like(tensor)
    #             stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

    #             del self.optimizer.state[group['params'][0]]
    #             group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
    #             self.optimizer.state[group['params'][0]] = stored_state

    #             optimizable_tensors[group["name"]] = group["params"][0]
    #     return optimizable_tensors
    

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        # List of parameter names that are point-specific and need pruning.
        # These names correspond to the keys in the self.optimizers dictionary.
        params_to_prune = ["points", "points_influ_scores", "points_scaler", "pc_feats"]
        if self.points_alpha is not None and self.args.geoms.alpha.use:
            params_to_prune.append("points_alpha")

        for name, optimizer in self.optimizers.items():
            if name not in params_to_prune:
                continue # Skip optimizers that do not manage point-specific parameters

            # Assuming each of these optimizers manages a single parameter group
            assert len(optimizer.param_groups) == 1, f"Optimizer for '{name}' should have only one parameter group"
            group = optimizer.param_groups[0]
            old_param = group['params'][0]

            # Get the optimizer state associated with the old parameter object
            stored_state = optimizer.state.get(old_param, None)

            # Create the new, pruned parameter from the data
            new_param = nn.Parameter(old_param.data[mask], requires_grad=old_param.requires_grad)

            # --- Correctly prune the gradient ---
            # If a gradient exists (i.e., after loss.backward()), it must also be pruned.
            if old_param.grad is not None:
                new_param.grad = old_param.grad[mask]

            if stored_state is not None:
                # Prune the state tensors (e.g., 'exp_avg', 'exp_avg_sq' for Adam)
                if 'exp_avg' in stored_state:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                if 'exp_avg_sq' in stored_state:
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                # Update the optimizer to track the new parameter and its state
                del optimizer.state[old_param]          # Remove state for the old parameter ID
                group['params'][0] = new_param          # Point the param_group to the new parameter object
                optimizer.state[new_param] = stored_state # Re-assign the pruned state to the new parameter ID
            else:
                # If optimizer has no state, just update the parameter
                group['params'][0] = new_param

            # Store the new parameter to be updated in the model
            optimizable_tensors[name] = new_param
            
        return optimizable_tensors
    
    
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for opt_name, optimizer in self.optimizers.items():
            if opt_name == name:
                group = optimizer.param_groups[0]
                old_param = group['params'][0]
                stored_state = optimizer.state.get(old_param, None)
                
                new_param = nn.Parameter(tensor.requires_grad_(True))
                if old_param.grad is not None:
                    # If the old parameter has a gradient, we need to create a new gradient for the new parameter.
                    new_param.grad = torch.zeros_like(old_param.grad)
                
                if stored_state is not None:
                    # For optimizers like Adam, we need to resize the state tensors.
                    if 'exp_avg' in stored_state:
                        stored_state["exp_avg"] = torch.zeros_like(tensor)
                    if 'exp_avg_sq' in stored_state:
                        stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                        
                    del optimizer.state[old_param]
                    group['params'][0] = new_param
                    optimizer.state[new_param] = stored_state
                else:
                    group['params'][0] = new_param

                optimizable_tensors[name] = new_param
        return optimizable_tensors
    

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for name, extension_tensor in tensors_dict.items():
            assert isinstance(extension_tensor, torch.Tensor), "Tensor {} is not a torch.Tensor".format(name)
            if name not in self.optimizers:
                raise ValueError("Optimizer for {} not found".format(name))
            assert len(self.optimizers[name].param_groups) == 1, "Optimizer for {} should have only one parameter group".format(name)
            
            optimizer = self.optimizers[name]
            group = optimizer.param_groups[0]
            old_param = group['params'][0]

            # --- 1. Create the new, larger parameter ---
            # Concatenate the data of the old parameter with the new extension tensor.
            new_param_data = torch.cat((old_param.data, extension_tensor), dim=0)
            new_param = nn.Parameter(new_param_data, requires_grad=old_param.requires_grad)

            # --- 2. Preserve and extend the gradient ---
            # This happens after loss.backward(), so old_param.grad exists.
            if old_param.grad is not None:
                # The new parts of the parameter have no gradient from the last backward pass.
                extension_grad = torch.zeros_like(extension_tensor)
                new_grad_data = torch.cat((old_param.grad, extension_grad), dim=0)
                new_param.grad = new_grad_data

            # --- 3. Update the optimizer's state (e.g., for Adam) ---
            stored_state = optimizer.state.get(old_param, None)
            if stored_state is not None:
                # For Adam, 'exp_avg' and 'exp_avg_sq' must be resized.
                if 'exp_avg' in stored_state:
                    extension_state = torch.zeros_like(extension_tensor)
                    stored_state['exp_avg'] = torch.cat((stored_state['exp_avg'], extension_state), dim=0)
                if 'exp_avg_sq' in stored_state:
                    extension_state = torch.zeros_like(extension_tensor)
                    stored_state['exp_avg_sq'] = torch.cat((stored_state['exp_avg_sq'], extension_state), dim=0)
                
                # --- 4. Update the optimizer to track the new parameter ---
                # This is the most critical step.
                del optimizer.state[old_param]          # Remove the state entry keyed by the old parameter's ID.
                group['params'][0] = new_param          # Update the parameter group to point to the new parameter object.
                optimizer.state[new_param] = stored_state # Add a new state entry keyed by the new parameter's ID.
            else:
                # If there's no state, just update the parameter group.
                group['params'][0] = new_param

            optimizable_tensors[name] = new_param
        return optimizable_tensors


    @torch.no_grad()
    def update_point_density(self):
        """
        Update per-point density using scipy cKDTree.
        Density is estimated as 1 / distance_to_kth_neighbor.
        """
        from scipy.spatial import cKDTree
        
        n_points = self.points.shape[0]
        
        if n_points < self.density_k:
            # Not enough points for KNN, set uniform density
            self.points_density.data.fill_(1.0)
            return
        
        device = self.points.device
        
        # Transfer points to CPU numpy
        points_np = (self.points.detach() / self.coord_scale).cpu().numpy()
        
        # Build KDTree and query KNN (parallel with workers=-1)
        tree = cKDTree(points_np)
        distances, _ = tree.query(points_np, k=self.density_k, workers=-1)
        
        # Compute density from k-th neighbor distance
        kth_distances = distances[:, -1]  # k-th neighbor (last column)
        kth_distances = np.maximum(kth_distances, 1e-10)
        density_np = 1.0 / kth_distances
        
        # Transfer result back to device
        self.points_density.data = torch.from_numpy(density_np).float().to(device)
        
        density = self.points_density
        print(f"[Density] Updated point density: min={density.min().item():.4f}, max={density.max().item():.4f}, "
              f"mean={density.mean().item():.4f}, std={density.std().item():.4f}")


    def prune_points(self, thresh, step=-1):
        num_pruned = 0
        if self.args.training.prune_by == "alpha" and (step < 7000 or step > 20000):
            prune_by = self.get_points_alpha
            if step > 20000:
                thresh = 0.01
            print("prune_by: points_alpha")
        else:
            prune_by = self.points_influ_scores
            # thresh = self.args.training.prune_thresh
            print("prune_by: points_influ_scores")
        
        if prune_by is not None:
            print("@@@@@@@@@  before prune: ", thresh, self.points.shape, prune_by.shape, prune_by.min().item(), prune_by.max().item())
            if self.args.training.prune_type == '<':
                mask = (prune_by[:, 0] > thresh)
            elif self.args.training.prune_type == '>':
                mask = (prune_by[:, 0] < thresh)
            num_to_prune = torch.sum(mask == 0)
            if num_to_prune > self.args.training.prune_max_num and step < self.args.training.prune_max_num_step:
                if self.args.training.prune_type == '<':
                    _, indices = torch.topk(prune_by[:, 0], self.args.training.prune_max_num, largest=False, sorted=False)
                elif self.args.training.prune_type == '>':
                    _, indices = torch.topk(prune_by[:, 0], self.args.training.prune_max_num, largest=True, sorted=False)
                mask = torch.ones_like(prune_by[:, 0], dtype=torch.bool)
                mask[indices] = False
                
            num_pruned = torch.sum(mask == 0).item()
            if num_pruned > 0:                
                optimizable_tensors = self._prune_optimizer(mask)
                self.points = optimizable_tensors["points"]
                self.points_influ_scores = optimizable_tensors["points_influ_scores"]
                if "points_scaler" in optimizable_tensors:
                    self.points_scaler = optimizable_tensors["points_scaler"]
                else:
                    # If points_scaler doesn't have an optimizer (e.g., requires_grad=False), prune it manually
                    self.points_scaler = nn.Parameter(self.points_scaler[mask], requires_grad=self.points_scaler.requires_grad)
                self.pc_feats = optimizable_tensors["pc_feats"]
                if self.points_alpha is not None:
                    self.points_alpha = optimizable_tensors["points_alpha"]

                self.points_last_grad = nn.Parameter(self.points_last_grad[mask, :], requires_grad=False)
                self.points_acc_grad = nn.Parameter(self.points_acc_grad[mask, :], requires_grad=False)
                self.points_acc_grad_norm = nn.Parameter(self.points_acc_grad_norm[mask], requires_grad=False)
                self.points_grad_cnt = nn.Parameter(self.points_grad_cnt[mask], requires_grad=False)
                self.points_density = nn.Parameter(self.points_density[mask], requires_grad=False)
            
            print("@@@@@@@@@  pruned {}/{}".format(num_pruned, mask.shape[0]))
            
            # Update point density after pruning
            if num_pruned > 0:
                self.update_point_density()
        return num_pruned


    def add_points(self, add_num, step=-1, prune_thresh=0.0):
        points = self.points.detach().cpu()
        point_features = None

        print("@@@@@@@@@  last_coord_grad: ", self.points_last_grad.shape, self.points_last_grad.min().item(), self.points_last_grad.max().item())
        print("@@@@@@@@@  acc_coord_grad: ", self.points_acc_grad.shape, self.points_acc_grad.min().item(), self.points_acc_grad.max().item())
        print("@@@@@@@@@  acc_coord_grad_norm: ", self.points_acc_grad_norm.shape, self.points_acc_grad_norm.min().item(), self.points_acc_grad_norm.max().item())
        print("@@@@@@@@@  grad_cnt: ", self.points_grad_cnt.shape, self.points_grad_cnt.min().item(), self.points_grad_cnt.max().item())
        
        acc_coord_grad = self.points_acc_grad.squeeze().detach().cpu().numpy()
        acc_coord_grad_norm = self.points_acc_grad_norm.squeeze().detach().cpu().numpy()
        grad_cnt = self.points_grad_cnt.squeeze().detach().cpu().numpy()

        quantiles = [0, 0.25, 0.5, 0.75, 1.0]

        # acc_coord_grad quantiles
        norm_acc_coord_grad = np.linalg.norm(acc_coord_grad, axis=1)
        norm_acc_coord_grad_q = np.quantile(norm_acc_coord_grad, quantiles)
        print("@@@@@@@ norm acc_coord_grad quantiles:")
        for i, q in enumerate(quantiles):
            print("  quantile {:.2f}: {:.6f}".format(q, norm_acc_coord_grad_q[i]))

        # acc_coord_grad_norm quantiles
        acc_coord_grad_norm_q = np.quantile(acc_coord_grad_norm, quantiles)
        print("@@@@@@@ acc_coord_grad_norm quantiles:")
        for i, q in enumerate(quantiles):
            print("  quantile {:.2f}: {:.6f}".format(q, acc_coord_grad_norm_q[i]))

        # grad_cnt quantiles
        grad_cnt_q = np.quantile(grad_cnt, quantiles)
        print("@@@@@@@ grad_cnt quantiles:")
        for i, q in enumerate(quantiles):
            print("  quantile {:.2f}: {:.6f}".format(q, grad_cnt_q[i]))

        # acc_coord_grad / grad_cnt quantiles
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_grad = np.divide(norm_acc_coord_grad, grad_cnt.squeeze(), out=np.zeros_like(norm_acc_coord_grad), where=grad_cnt.squeeze() != 0)
            ratio_grad_norm = np.divide(acc_coord_grad_norm, grad_cnt.squeeze(), out=np.zeros_like(acc_coord_grad_norm), where=grad_cnt.squeeze() != 0)

        ratio_grad_q = np.quantile(ratio_grad, quantiles)
        print("@@@@@@@ acc_coord_grad / grad_cnt quantiles:")
        for i, q in enumerate(quantiles):
            print("  quantile {:.2f}: {}".format(q, ratio_grad_q[i]))

        ratio_grad_norm_q = np.quantile(ratio_grad_norm, quantiles)
        print("@@@@@@@ acc_coord_grad_norm / grad_cnt quantiles:")
        for i, q in enumerate(quantiles):
            print("  quantile {:.2f}: {:.6f}".format(q, ratio_grad_norm_q[i]))

        point_features = self.pc_feats.detach().cpu()       
        point_alphas = self.points_alpha.detach().cpu() if self.points_alpha is not None else None 
        point_scalers = self.points_scaler.detach().cpu()
        new_points, num_new_points, new_influ_scores, new_point_features, new_alphas, new_scalers, to_add_inds = add_points_knn(points, self.points_influ_scores.detach().cpu(), add_num=add_num,
                                                                                                                    k=self.args.geoms.points.add_k, comb_type=self.args.geoms.points.add_type,
                                                                                                                    sample_k=self.args.geoms.points.add_sample_k, sample_type=self.args.geoms.points.add_sample_type,
                                                                                                                    point_features=point_features, point_alphas=point_alphas, point_scalers=point_scalers,
                                                                                                                    last_coord_grad=self.points_last_grad.detach().cpu(), 
                                                                                                                    acc_coord_grad=self.points_acc_grad.detach().cpu(), 
                                                                                                                    acc_coord_grad_norm=self.points_acc_grad_norm.detach().cpu(),
                                                                                                                    grad_cnt=self.points_grad_cnt.detach().cpu(),
                                                                                                                    move_scale=self.args.geoms.points.move_scale)

        if num_new_points > 0:
            # Convert new_scalers from numpy to torch if needed
            if isinstance(new_scalers, np.ndarray):
                new_scalers = torch.from_numpy(new_scalers).float()
            elif new_scalers is None:
                # Fallback: initialize to default value if not computed
                scaler_init_val = getattr(self.args.geoms.points, 'scaler_init_val', 1.0)
                new_scalers = torch.ones(num_new_points, 1) * scaler_init_val
            
            if self.args.geoms.points.save_added_points:
                save_dir = os.path.join(self.args.save_dir, self.args.index, "point_clouds")
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, "added_points_{}.ply".format(step))
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points.detach().cpu().numpy())
                colors = np.ones_like(points.detach().cpu().numpy()) * 0.5
                colors[to_add_inds, :] = np.array([1, 0, 0])
                pcd.colors = o3d.utility.Vector3dVector(colors)
                o3d.io.write_point_cloud(save_path, pcd)
                print("@@@@@@@@@  saved added points to {}".format(save_path))
            
            d = {
                "points": new_points.to(self.points.device),
                "points_influ_scores": new_influ_scores.to(self.points_influ_scores.device),
                "pc_feats": new_point_features.to(self.pc_feats.device),
            }
            # Only include points_scaler in d if it has an optimizer
            points_scaler_new = new_scalers.to(self.points_scaler.device)
            if "points_scaler" in self.optimizers:
                d["points_scaler"] = points_scaler_new
            if self.points_alpha is not None:
                d["points_alpha"] = new_alphas.to(self.points_alpha.device)
            optimizable_tensors = self.cat_tensors_to_optimizer(d)
            self.points = optimizable_tensors["points"]
            self.points_influ_scores = optimizable_tensors["points_influ_scores"]
            if "points_scaler" in optimizable_tensors:
                self.points_scaler = optimizable_tensors["points_scaler"]
            else:
                # If points_scaler doesn't have an optimizer, concatenate it manually
                self.points_scaler = nn.Parameter(torch.cat([self.points_scaler, points_scaler_new], dim=0), requires_grad=self.points_scaler.requires_grad)
            self.pc_feats = optimizable_tensors["pc_feats"]
            if self.points_alpha is not None:
                self.points_alpha = optimizable_tensors["points_alpha"]
            
            self.points_last_grad = nn.Parameter(torch.zeros(self.points.shape[0], 3, device=self.points.device), requires_grad=False)
            self.points_acc_grad = nn.Parameter(torch.zeros(self.points.shape[0], 3, device=self.points.device), requires_grad=False)
            self.points_acc_grad_norm = nn.Parameter(torch.zeros(self.points.shape[0], device=self.points.device), requires_grad=False)
            self.points_grad_cnt = nn.Parameter(torch.zeros(self.points.shape[0], device=self.points.device), requires_grad=False)
            # Reinitialize points_density for new point count and update
            self.points_density = nn.Parameter(torch.ones(self.points.shape[0], device=self.points.device), requires_grad=False)
            print("@@@@@@@@@  added {} points".format(num_new_points))
            
            # Update point density after adding points
            self.update_point_density()

        return num_new_points
        

    def step(self, step=-1, grad_clip_norm=None, grad_clip_value=None):
        # Update point density every T iterations
        if self.density_update_interval > 0 and step > 0 and step % self.density_update_interval == 0:
            self.update_point_density()
        
        if self.points.grad is not None:            
            self.points_last_grad.data = self.points.grad
            self.points_acc_grad.data += self.points.grad
            self.points_acc_grad_norm.data += torch.norm(self.points.grad, dim=-1)
            self.points_grad_cnt.data += (self.points.grad.abs().sum(-1) != 0).float()

            # Add noise perturbation to gradient based on point density
            # noise * (1 / density): sparse areas get more noise, dense areas get less
            if self.grad_noise_std > 0:
                noise = torch.randn_like(self.points.grad) * self.grad_noise_std
                # Scale noise inversely by density (add small epsilon to avoid division by zero)
                inv_density = 1.0 / (self.points_density.unsqueeze(-1) + 1e-8)
                self.points.grad.data += noise * inv_density
        
        # self.points_influ_scores.grad[self.points_influ_scores >= self.args.geoms.points.influ_max] = 0
        # self.points_influ_scores.grad[self.points_influ_scores <= self.args.geoms.points.influ_min] = 0

        if (
            self.args.models.attn.use_pc_feats_directly
            and self.args.models.attn.use_sh
            # Phase 2 freezes pc_feats, so it has no gradient to rescale.
            and self.pc_feats.grad is not None
        ):
            self.pc_feats.grad[:, 1:, :] /= self.args.models.attn.sh_N_factor
        
        clip_norm_on = grad_clip_norm is not None and grad_clip_norm > 0
        clip_value_on = grad_clip_value is not None and grad_clip_value > 0
        if (clip_norm_on or clip_value_on) and self.scaler.is_enabled():
            # Clip real gradients, not loss-scaled ones.  Without this the
            # thresholds are compared against s*|g| (s starts at 65536), so
            # clipping fires on every iteration and shrinks the applied update
            # by the same factor.  `scaler.step()` below skips its own unscale
            # for any optimizer already unscaled in this iteration.
            for opt_name, optimizer in self.optimizers.items():
                if optimizer is None:
                    continue
                if opt_name == "topk_mlp" and step < self.args.models.topk_mlp.start_step:
                    continue
                if any(
                    param.grad is not None
                    for param_group in optimizer.param_groups
                    for param in param_group["params"]
                ):
                    self.scaler.unscale_(optimizer)

        # Apply gradient clipping if specified
        if grad_clip_norm is not None and grad_clip_norm > 0:
            # Collect all parameters from optimizers - these are the parameters actually being optimized
            all_params = []
            for optimizer in self.optimizers.values():
                if optimizer is not None:
                    for param_group in optimizer.param_groups:
                        all_params.extend(param_group['params'])
            
            total_norm = torch.nn.utils.clip_grad_norm_(all_params, grad_clip_norm)
            
            if step % 201 == 0:
                print(f"Gradient clipping applied: norm {total_norm:.4f} (clip_norm={grad_clip_norm})")
        
        if grad_clip_value is not None and grad_clip_value > 0:
            # Collect all parameters from optimizers - these are the parameters actually being optimized
            all_params = []
            for optimizer in self.optimizers.values():
                if optimizer is not None:
                    for param_group in optimizer.param_groups:
                        all_params.extend(param_group['params'])
                        
            torch.nn.utils.clip_grad_value_(all_params, grad_clip_value)
            
            if step % 201 == 0:
                print(f"Gradient clipping applied: value {grad_clip_value:.4f} (clip_value={grad_clip_value})")

        any_stepped = False
        for opt_name, optimizer in self.optimizers.items():
            if opt_name == "topk_mlp" and step < self.args.models.topk_mlp.start_step:
                continue
            if optimizer is not None:
                # print("@@@@@@@@@  step optimizer: ", opt_name)
                has_grad = any(
                    param.grad is not None and param.grad.data.numel() > 0
                    for param_group in optimizer.param_groups
                    for param in param_group['params']
                    if param.grad is not None
                )
                if has_grad:
                    self.scaler.step(optimizer)
                    any_stepped = True

        for sch_name, scheduler in self.schedulers.items():
            if scheduler is not None:
                scheduler.step()

        self.attn_lr = 0
        if 'attn_v' in self.optimizers:
            if self.schedulers['attn_v'] is not None:
                self.attn_lr = self.schedulers['attn_v'].get_last_lr()[0]
            else:
                self.attn_lr = self.optimizers['attn_v'].param_groups[0]['lr']
        elif 'attn_other' in self.optimizers:
            if self.schedulers['attn_other'] is not None:
                self.attn_lr = self.schedulers['attn_other'].get_last_lr()[0]
            else:
                self.attn_lr = self.optimizers['attn_other'].param_groups[0]['lr']
        elif 'attn' in self.optimizers:  # Backward compatibility
            if self.schedulers['attn'] is not None:
                self.attn_lr = self.schedulers['attn'].get_last_lr()[0]
            else:
                self.attn_lr = self.optimizers['attn'].param_groups[0]['lr']

        self.pts_lr = 0
        if 'points' in self.optimizers:
            if self.schedulers['points'] is not None:
                self.pts_lr = self.schedulers['points'].get_last_lr()[0]
            else:
                self.pts_lr = self.optimizers['points'].param_groups[0]['lr']
        
        return any_stepped


    def get_attn_weights(self, scores):
        if self.args.attn_act == "softmax":
            if self.args.models.attn.append_bkg_points and self.bkg_exp_scaler is not None:
                # Custom softmax with learnable scaler for background point
                # scores shape: (..., K+1, 1) where K+1 includes bkg point at the last position
                # Formula: attn_i = exp(s_i) / (sum_{j=1}^{K} exp(s_j) + c * exp(s_bkg))
                #          attn_bkg = c * exp(s_bkg) / (sum_{j=1}^{K} exp(s_j) + c * exp(s_bkg))
                scaled_scores = scores * self.args.attn_act_temp
                # For numerical stability, subtract max
                max_scores = torch.max(scaled_scores, dim=-2, keepdim=True).values
                exp_scores = torch.exp(scaled_scores - max_scores)
                
                # Scale the background point's exp score (last element in dim=-2)
                # exp_scores shape: (..., K+1, 1)
                fg_exp_scores = exp_scores[..., :-1, :]  # (..., K, 1)
                bkg_exp_score = exp_scores[..., -1:, :]  # (..., 1, 1)
                scaled_bkg_exp_score = bkg_exp_score * self.bkg_exp_scaler
                
                # Compute normalization: sum of fg + scaled bkg
                sum_exp = torch.sum(fg_exp_scores, dim=-2, keepdim=True) + scaled_bkg_exp_score
                
                # Compute attention weights
                fg_attn = fg_exp_scores / (sum_exp + self.eps)
                bkg_attn = scaled_bkg_exp_score / (sum_exp + self.eps)
                attn = torch.cat([fg_attn, bkg_attn], dim=-2)
            else:
                attn = F.softmax(scores * self.args.attn_act_temp, dim=-2)
        elif self.args.attn_act == "minmax":
            min_value = scores.min(dim=-2, keepdim=True)[0]
            max_value = scores.max(dim=-2, keepdim=True)[0]
            attn = (scores - min_value) / (max_value - min_value + 1e-6)
        elif self.args.attn_act == "sum":
            attn = scores / torch.sum(scores, dim=-2, keepdim=True)
        elif self.args.attn_act == "softpick":
            # print("scores", scores.shape, scores.min().item(), scores.max().item())
            x_m = torch.max(scores, dim=-2, keepdim=True).values
            # print("x_m", x_m.shape, x_m.min().item(), x_m.max().item())
            x_m_e_m = torch.exp(-x_m)
            x_e_1 = torch.exp(scores - x_m) - x_m_e_m
            r_x_e_1 = F.relu(x_e_1)
            a_x_e_1 = torch.where(scores.isfinite(), torch.abs(x_e_1), 0)
            attn = r_x_e_1 / (torch.sum(a_x_e_1, dim=-2, keepdim=True) + self.eps)
            # print("attn", attn.shape, attn.min().item(), attn.max().item(), attn.mean().item())
            # exit(0)
        else:
            raise ValueError("Unknown attention activation: {}".format(self.args.attn_act))
        return attn
    
    def get_scheduled_weight(self, base_weight, step, start, stop, schedule_type="constant"):
        """Return a possibly time-varying weight within [start, stop].

        schedule_type:
          - "constant":       base_weight within window, 0 outside
          - "cosine":         cosine decay from base_weight -> 0 over window (alias of cosine_decay)
          - "cosine_decay":   cosine decay from base_weight -> 0 over window
          - "cosine_warmup":  cosine warm-up from 0 -> base_weight over window
        """
        if base_weight <= 0:
            return 0.0
        if step < start or step > stop:
            return 0.0
        T = max(float(stop - start), 1.0)
        t = (step - start) / T  # normalized progress in [0,1]
        if schedule_type in ("cosine", "cosine_decay"):
            # Cosine decay: 1 -> 0
            ramp = 0.5 * (1.0 + torch.cos(torch.tensor(t) * torch.pi))
            return float(base_weight * ramp.item())
        if schedule_type == "cosine_warmup":
            # Cosine warmup: 0 -> 1
            ramp = 0.5 * (1.0 - torch.cos(torch.tensor(t) * torch.pi))
            return float(base_weight * ramp.item())
        # default constant
        return float(base_weight)

    def compute_scheduled_loss(self, loss_name, cur_step, loss_fn, extra_condition=True):
        """Compute a regularizer loss with scheduler support.
        
        Args:
            loss_name: Name of the loss (e.g., 'point_on_ray_loss')
            cur_step: Current training step
            loss_fn: Callable that returns the loss tensor (only called if weight > 0)
            extra_condition: Additional condition that must be True for loss to be computed
            
        Returns:
            Tuple of (loss_tensor, loss_weight)
        """
        schedule = self.args.training.get(f'{loss_name}_schedule', 
                                          self.args.training.get('regularizer_schedule', 'constant'))
        weight = self.get_scheduled_weight(
            self.args.training.get(f'{loss_name}_weight', 0.0),
            cur_step,
            self.args.training.get(f'{loss_name}_start', -1e9),
            self.args.training.get(f'{loss_name}_stop', 1e9),
            schedule,
        )
        if weight > 0 and extra_condition:
            loss = loss_fn()
        else:
            loss = torch.zeros(0, device=self.device).sum()
        return loss, weight
    
    def get_drop_points_max_ratio(self, cur_step=-1):
        """Get the current max dropout ratio based on the scheduler.
        
        Returns the maximum ratio that can be used for dropout, which decays
        from drop_points_max_ratio to 0 during training based on the schedule type.
        
        Args:
            cur_step: Current training step (-1 if not available)
            
        Returns:
            Current maximum dropout ratio (0.0 if scheduling is disabled or step is invalid)
        """
        max_ratio = getattr(self.args.geoms.points, 'drop_points_max_ratio', -1.0)
        if max_ratio <= 0:
            return 0.0
        
        # Check if we're before the dropout start step (no dropout before this step)
        drop_start_step = getattr(self.args.geoms.points, 'drop_points_start_step', 0)
        if cur_step >= 0 and cur_step < drop_start_step:
            return 0.0
        
        if cur_step < 0:
            # If step is not available, return the base ratio
            return max_ratio
        
        schedule_type = getattr(self.args.geoms.points, 'drop_points_max_ratio_schedule_type', 'none')
        if schedule_type == 'none':
            return max_ratio
        
        start_step = getattr(self.args.geoms.points, 'drop_points_max_ratio_start_step', 0)
        end_step = getattr(self.args.geoms.points, 'drop_points_max_ratio_end_step', -1)
        if end_step < 0:
            end_step = self.args.training.steps
        
        if cur_step < start_step:
            return max_ratio
        if cur_step >= end_step:
            return 0.0
        
        T = max(float(end_step - start_step), 1.0)
        t = (cur_step - start_step) / T  # normalized progress in [0,1]
        
        if schedule_type == 'cosine':
            # Cosine decay: 1 -> 0
            ramp = 0.5 * (1.0 + np.cos(t * np.pi))
            return float(max_ratio * ramp)
        elif schedule_type == 'linear':
            # Linear decay: 1 -> 0
            ramp = 1.0 - t
            return float(max_ratio * ramp)
        else:
            # Unknown schedule type, return base ratio
            return max_ratio
    
    def get_topk_mlp_loss(self, step):
        N, H, W, K, _ = self.proximity_attn.pd.shape
        topk_loss = torch.zeros(0, device=self.device).sum()
        if self.args.geoms.points.select_k_type == "mlp1":
            d2r, pd = self.proximity_attn.d2r.detach().squeeze(-1), self.proximity_attn.pd.detach().squeeze(-1)
            act_scores = self.proximity_attn.act_scores.detach().reshape(N, H, W, -1)
            topk_loss = self.topk_mlp([d2r, pd], act_scores, step=step)
            
        elif self.args.geoms.points.select_k_type == "mlp2":
            d2r, pd = self.proximity_attn.d2r.detach().squeeze(-1), self.proximity_attn.pd.detach().squeeze(-1)
            act_scores = self.proximity_attn.act_scores.detach().reshape(N, H, W, -1)
            max_pd = pd.max(dim=-1, keepdim=True)[0]
            topk_loss = self.topk_mlp([d2r, (max_pd - pd + self.eps).pow(self.args.geoms.points.select_k_cone_pow)], act_scores, step=step)

        elif self.args.geoms.points.select_k_type == "mlp3":
            d2r, pd = self.proximity_attn.d2r.detach().squeeze(-1), self.proximity_attn.pd.detach().squeeze(-1)
            act_scores = self.proximity_attn.act_scores.detach().reshape(N, H, W, -1)
            min_pd = pd.min(dim=-1, keepdim=True)[0]
            topk_loss = self.topk_mlp([d2r, (pd - min_pd + 1).pow(self.args.geoms.points.select_k_cone_pow)], act_scores, step=step)

        elif self.args.geoms.points.select_k_type == "mlp4":
            d2r, pd = self.proximity_attn.d2r.detach().squeeze(-1), self.proximity_attn.pd.detach().squeeze(-1)
            act_scores = self.proximity_attn.act_scores.detach().reshape(N, H, W, -1)
            topk_loss = self.topk_mlp([d2r, pd, 1 / (pd + self.eps)], act_scores, step=step)

        elif self.args.geoms.points.select_k_type == "mlp5":
            d2r, pd = self.proximity_attn.d2r.detach().squeeze(-1), self.proximity_attn.pd.detach().squeeze(-1)
            act_scores = self.proximity_attn.act_scores.detach().reshape(N, H, W, -1)
            max_pd = pd.max(dim=-1, keepdim=True)[0]
            topk_loss = self.topk_mlp([d2r, pd, max_pd - pd, 1 / (pd + self.eps), 1 / (max_pd - pd + self.eps)], act_scores, step=step)
        
        elif self.args.geoms.points.select_k_type == "mlp6":
            d2r, pd = self.proximity_attn.d2r.detach().squeeze(-1), self.proximity_attn.pd.detach().squeeze(-1)
            act_scores = self.proximity_attn.act_scores.detach().reshape(N, H, W, -1)
            max_pd = pd.max(dim=-1, keepdim=True)[0]
            topk_loss = self.topk_mlp([d2r, pd, 1 / (max_pd - pd + self.eps)], act_scores, step=step)

        elif self.args.geoms.points.select_k_type == "mlp7":
            d2r, pd = self.proximity_attn.d2r.detach().squeeze(-1), self.proximity_attn.pd.detach().squeeze(-1)
            act_scores = self.proximity_attn.act_scores.detach().reshape(N, H, W, -1)
            max_pd = pd.max(dim=-1, keepdim=True)[0]
            min_pd = pd.min(dim=-1, keepdim=True)[0]
            topk_loss = self.topk_mlp([d2r, pd, max_pd - pd, 1 / (pd + self.eps), 1 / (max_pd - pd + self.eps), 
                                        1 / (pd + self.eps) ** 2, 1 / (max_pd - pd + self.eps) ** 2], act_scores, step=step)
        
        elif self.args.geoms.points.select_k_type == "mlp8":
            d2r, pd = self.proximity_attn.d2r.detach().squeeze(-1), self.proximity_attn.pd.detach().squeeze(-1)
            act_scores = self.proximity_attn.act_scores.detach().reshape(N, H, W, -1)
            max_pd = pd.max(dim=-1, keepdim=True)[0]
            min_pd = pd.min(dim=-1, keepdim=True)[0]
            topk_loss = self.topk_mlp([d2r, pd, max_pd - pd, d2r ** 2, pd ** 2, (max_pd - pd) ** 2], act_scores, step=step)
            
        return topk_loss
    
    
    def get_alpha_attn_loss(self, attn):
        if self.points_alpha is None:
            return torch.zeros(0, device=self.device).sum()
        selected_alpha = self.get_points_alpha[self.select_k_ind]
        topk_attn = attn[..., :-1, :]
        bkg_attn = attn[..., -1:, :]
        if self.args.geoms.alpha.normalize_topk_attn:
            topk_attn = topk_attn / (1 - bkg_attn + self.args.models.topk_eps)
        if self.args.geoms.alpha.detach_attn:
            topk_attn = topk_attn.detach()
        loss = self.alpha_attn_loss(selected_alpha, topk_attn)
        return loss


    @torch.no_grad()
    def evaluate(self, rays_o, rays_d, c2w, pix_coords, step=-1, shading_code=None, surface_points=None, normals=None, deformed_points=None, rays_d_no_norm=None):
        if self.args.geoms.points.select_k_rnd:
            self.select_k = self.args.geoms.points.select_k_max
            
        points = self.get_points(deformed_points, cur_step=step)
        pc_feats = self.get_pc_feats()
        cur_points_influ_score = self.get_influ_scores()
        cur_points_scaler = self.get_scaler()
            
        points_2d, points_cam = fused_projection(points, c2w, self.fx, self.fy, self.cx, self.cy, self.H, self.W)
        z = points_cam[..., 2]

        if self.args.rescale_bkg_points_for_attn:
            points_for_attn = points.clone()
            points_for_attn[self.bkg_points_mask == 0] /= self.args.rescale_bkg_points_for_attn_scale
        else:
            points_for_attn = points
        select_k_ind, min_d2r = self._get_topk_point_inds(rays_o, rays_d, points, pix_coords, points_2d, z, step, surface_points=surface_points, normals=normals, rays_d_no_norm=rays_d_no_norm)

        append_bkg_points_feats = None
        if self.args.models.attn.append_bkg_points:
            # Check if we should use cubemap feature map for background point features
            if self.cubemap_use_feature_map:
                # Sample features from cubemap based on ray-sphere intersection directions
                cubemap_feats = self.get_cubemap_background_features(rays_o, rays_d, step)
                if cubemap_feats is not None:
                    # Shape: (N, H, W, feature_dim) -> (N, H, W, 1, feature_dim)
                    append_bkg_points_feats = cubemap_feats.unsqueeze(-2)
                elif self.append_bkg_points_feats is not None:
                    # Fallback to learned features if cubemap not active yet
                    append_bkg_points_feats = self.append_bkg_points_feats
            elif self.append_bkg_points_feats is not None:
                if self.append_bkg_points_feats.shape[0] == 4:
                    append_bkg_points_feats = torch.sum(self.append_bkg_points_feats, dim=0, keepdim=True)
                else:
                    append_bkg_points_feats = self.append_bkg_points_feats
        
        # Get cubemap depth offset for background surface adjustment
        bkg_depth_offset = self.get_cubemap_depth_offset(rays_o, rays_d, cur_step=step)
        
        scores, embedv, select_k_ind, reg_loss, hit_pred = self.proximity_attn(rays_o, rays_d, points_for_attn, pc_feats, cur_points_influ_score, select_k_ind, z, c2w, self.bkg_token, step, 
                                                                            self.args.training.steps, evaluate=True, append_bkg_points_feats=append_bkg_points_feats, points_scaler=cur_points_scaler,
                                                                            bkg_points_mask=self.bkg_points_mask, bkg_depth_offset=bkg_depth_offset)
        self.select_k_ind = select_k_ind

        N, H, W, _ = rays_d.shape
        embedv = embedv.reshape(N, H, W, -1, embedv.shape[-1])
        scores = scores.reshape(N, H, W, -1, 1)

        cur_points_influ_score = cur_points_influ_score[select_k_ind] if cur_points_influ_score is not None else None
        if cur_points_influ_score is not None:
            if self.args.influ_scores_fuse_type == "multiply":
                if self.bkg_token is not None or self.args.models.attn.append_bkg_points:
                    cur_points_influ_score = torch.cat([cur_points_influ_score, torch.ones(N, H, W, 1, 1, device=self.device)], dim=-2)
                scores = scores * cur_points_influ_score
            elif self.args.influ_scores_fuse_type == "add":
                if self.bkg_token is not None or self.args.models.attn.append_bkg_points:
                    cur_points_influ_score = torch.cat([cur_points_influ_score, torch.zeros(N, H, W, 1, 1, device=self.device)], dim=-2)
                scores = scores + cur_points_influ_score
            else:
                raise ValueError("Unknown influence scores fusion type: {}".format(self.args.influ_scores_fuse_type))

        if self.args.geoms.no_additional_bkg:
            # No special background treatment - use all attention weights directly
            attn = self.get_attn_weights(scores)
            self.last_embedv = embedv
            fused_features = torch.sum(embedv * attn, dim=3, keepdim=True)   # (N, H, W, 1, C)
        elif self.bkg_feats is not None and not self.args.geoms.points.use_bkg_points and not self.args.models.attn.append_bkg_points:
            bkg_seq_len = self.bkg_feats.shape[0]
            if self.bkg_token is None:
                scores = torch.cat([scores, self.bkg_score.expand(N, H, W, bkg_seq_len, -1)], dim=-2)
            # attn = F.softmax(scores, dim=3) # (N, H, W, num_pts+bkg_seq_len, 1)
            attn = self.get_attn_weights(scores)
            topk_attn = attn[..., :-1, :]
            bkg_attn = attn[..., -1:, :]
            if self.args.geoms.background.use_dumb_constant:
                bkg_attn = (bkg_attn * self.dumb_constant).clamp(0, 1)
            self.raw_topk_attn = topk_attn
            if self.args.models.normalize_topk_attn:
                topk_attn = topk_attn / (1 - bkg_attn + self.args.models.topk_eps)
            # Store embedv for later access
            self.last_embedv = embedv
            fused_features = torch.sum(embedv * topk_attn, dim=3, keepdim=True)   # (N, H, W, 1, C)
        else:
            # attn = F.softmax(scores, dim=3)
            attn = self.get_attn_weights(scores)
            
            if self.args.geoms.points.use_bkg_points and self.args.geoms.points.bkg_points_given_color:
                selected_bkg_points_mask = self.bkg_points_mask[select_k_ind].expand(-1, -1, -1, -1, 3)
                embedv = torch.where(selected_bkg_points_mask == 1., embedv, self.bkg_feats.expand(N, H, W, self.select_k, -1))
                
            if self.args.geoms.points.use_bkg_points and self.args.geoms.points.bkg_points_given_embedv:
                selected_bkg_points_mask = self.bkg_points_mask[select_k_ind].expand(-1, -1, -1, -1, 3)
                embedv = torch.where(selected_bkg_points_mask == 1., embedv, self.bkg_points_embedv.expand_as(embedv))
                
            if self.args.models.attn.append_bkg_points and getattr(self.args.models.attn, 'append_bkg_points_use_embedv', False) and self.append_bkg_points_embedv is not None:
                # Create a mask for the last element (appended background point) and use torch.where to avoid inplace operation
                append_bkg_mask = torch.zeros(embedv.shape[:-1], dtype=torch.bool, device=embedv.device)
                append_bkg_mask[..., -1] = True
                if self.append_bkg_points_embedv.ndim == 2:
                    if self.append_bkg_points_embedv.shape[0] == 4:
                        append_bkg_points_embedv = torch.sum(self.append_bkg_points_embedv, dim=0)
                    else:
                        append_bkg_points_embedv = self.append_bkg_points_embedv.squeeze(0)
                else:
                    append_bkg_points_embedv = self.append_bkg_points_embedv
                append_bkg_embedv = append_bkg_points_embedv.expand(N, H, W, -1).unsqueeze(-2)
                embedv = torch.where(append_bkg_mask.unsqueeze(-1), append_bkg_embedv, embedv)
            
            # Store embedv for later access (needed for foreground/background separation with cubemap_feature_textures)
            self.last_embedv = embedv
                
            fused_features = torch.sum(embedv * attn, dim=3, keepdim=True)   # (N, H, W, 1, C)

        return fused_features, attn, min_d2r, hit_pred


    def forward(self, rays_o, rays_d, c2w, pix_coords, pixels, bkg_color, mask, cur_step=-1, shading_code=None, surface_points=None, normals=None, deformed_points=None, rays_d_no_norm=None, log=True, drop=True):
        bkg_reg_loss = torch.zeros(0, device=self.device).sum()
        
        gamma, beta = None, None
        if shading_code is not None and self.mapping_mlp is not None:
            affine = self.mapping_mlp(shading_code)
            affine_dim = affine.shape[-1]
            gamma, beta = affine[:affine_dim//2], affine[affine_dim//2:]
        
            if cur_step % 200 == 0:
                print(shading_code.min().item(), shading_code.max().item(), gamma.min().item(), gamma.max().item(), beta.min().item(), beta.max().item())
        
        if self.args.geoms.points.select_k_rnd:
            self.select_k = random.randint(self.args.geoms.points.select_k, self.args.geoms.points.select_k_max)

        grid = None
        if self.args.geoms.rays.perturb:
            rays_d, grid, pix_coords = self.perturb_rays(rays_o, pixels, c2w, pix_coords)
            
        points = self.get_points(deformed_points, drop=self.args.geoms.points.drop_points_max_ratio > 0 and drop, cur_step=cur_step)
        if self.args.geoms.points.add_noise:
            points = self.add_gaussian_noise_to_points(points, cur_step)
        pc_feats = self.get_pc_feats(bkg_color)
        cur_points_influ_score = self.get_influ_scores()
        cur_points_scaler = self.get_scaler()

        points_2d, points_cam = fused_projection(points, c2w, self.fx, self.fy, self.cx, self.cy, self.H, self.W)
        z = points_cam[..., 2]

        if self.args.rescale_bkg_points_for_attn:
            points_for_attn = points.clone()
            points_for_attn[self.bkg_points_mask == 0] /= self.args.rescale_bkg_points_for_attn_scale
        else:
            points_for_attn = points

        append_bkg_points_feats = None
        if self.args.models.attn.append_bkg_points:
            # Check if we should use cubemap feature map for background point features
            if self.cubemap_use_feature_map:
                # Sample features from cubemap based on ray-sphere intersection directions
                cubemap_feats = self.get_cubemap_background_features(rays_o, rays_d, cur_step)
                if cubemap_feats is not None:
                    # Shape: (N, H, W, feature_dim) -> (N, H, W, 1, feature_dim)
                    append_bkg_points_feats = cubemap_feats.unsqueeze(-2)
                    if cur_step >= 0 and cur_step % 200 == 0:
                        print(' Using cubemap feature map for append_bkg_points_feats, shape:', append_bkg_points_feats.shape)
                elif self.append_bkg_points_feats is not None:
                    # Fallback to learned features if cubemap not active yet
                    append_bkg_points_feats = self.append_bkg_points_feats
            elif self.append_bkg_points_feats is not None:
                # Original behavior: use learned shared features
                if self.append_bkg_points_feats.shape[0] == 4:
                    weights = torch.ones(4, device=self.device)
                    if bkg_color is not None:
                        weights[1:] = bkg_color.squeeze()
                    append_bkg_points_feats = torch.sum(self.append_bkg_points_feats * weights.unsqueeze(-1), dim=0, keepdim=True)
                else:
                    append_bkg_points_feats = self.append_bkg_points_feats

        select_k_ind, min_d2r = self._get_topk_point_inds(rays_o, rays_d, points, pix_coords, points_2d, z, cur_step, surface_points=surface_points, normals=normals, rays_d_no_norm=rays_d_no_norm)
        
        # Get cubemap depth offset for background surface adjustment
        bkg_depth_offset = self.get_cubemap_depth_offset(rays_o, rays_d, cur_step=cur_step)
        
        scores, embedv, select_k_ind, prop_reg_loss, hit_pred = self.proximity_attn(rays_o, rays_d, points_for_attn, pc_feats, cur_points_influ_score, select_k_ind, z, c2w, 
                                                                                    self.bkg_token, cur_step, self.args.training.steps, append_bkg_points_feats=append_bkg_points_feats, points_scaler=cur_points_scaler,
                                                                                    bkg_points_mask=self.bkg_points_mask, bkg_depth_offset=bkg_depth_offset)
        self.select_k_ind = select_k_ind

        if log and cur_step >= 0 and cur_step % 200 == 0:
            print(' select k:', cur_step, self.select_k)
            print(' embedv:', cur_step, embedv.shape, embedv.min().item(), embedv.max().item(), embedv.mean().item(), embedv.std().item())
            print(' scores bf:', cur_step, scores.shape, scores.min().item(), scores.max().item(), scores.mean().item(), scores.std().item())
            print(' prop_reg_loss:', cur_step, prop_reg_loss.item())
            print(' bkg score:', cur_step, self.bkg_score.shape, self.bkg_score.mean().item())
            print(' points_scaler:', cur_step, self.points_scaler.shape, self.points_scaler.min().item(), self.points_scaler.max().item(), self.points_scaler.mean().item(), self.points_scaler.std().item())
            if self.bkg_token is not None:
                print(' bkg_token:', cur_step, self.bkg_token.shape, self.bkg_token.min().item(), self.bkg_token.max().item(), self.bkg_token.mean().item(), self.bkg_token.std().item())
            if self.bkg_points_influ_scores is not None:
                print(' bkg_points_influ_scores:', cur_step, self.bkg_points_influ_scores.shape, self.bkg_points_influ_scores.min().item(), self.bkg_points_influ_scores.max().item(), self.bkg_points_influ_scores.mean().item())
            if self.bkg_points_pc_feats is not None:
                print(' bkg_points_pc_feats:', cur_step, self.bkg_points_pc_feats.shape, self.bkg_points_pc_feats.min().item(), self.bkg_points_pc_feats.max().item(), self.bkg_points_pc_feats.mean().item())
            if self.bkg_exp_scaler is not None:
                print(' bkg_exp_scaler:', cur_step, self.bkg_exp_scaler.shape, self.bkg_exp_scaler.item())

        N, H, W, _ = rays_d.shape
        embedv = embedv.reshape(N, H, W, -1, embedv.shape[-1])
        scores = scores.reshape(N, H, W, -1, 1)

        cur_points_influ_score = cur_points_influ_score[select_k_ind] if cur_points_influ_score is not None else None
        if cur_points_influ_score is not None:
            if self.args.influ_scores_fuse_type == "multiply":
                if self.bkg_token is not None or self.args.models.attn.append_bkg_points:
                    cur_points_influ_score = torch.cat([cur_points_influ_score, torch.ones(N, H, W, 1, 1, device=self.device)], dim=-2)
                scores = scores * cur_points_influ_score
            elif self.args.influ_scores_fuse_type == "add":
                if self.bkg_token is not None or self.args.models.attn.append_bkg_points:
                    cur_points_influ_score = torch.cat([cur_points_influ_score, torch.zeros(N, H, W, 1, 1, device=self.device)], dim=-2)
                scores = scores + cur_points_influ_score
            else:
                raise ValueError("Unknown influence scores fusion type: {}".format(self.args.influ_scores_fuse_type))

        int_rgb = None
        if self.args.geoms.no_additional_bkg:
            # No special background treatment - use all attention weights directly
            attn = self.get_attn_weights(scores)
            fused_features = torch.sum(embedv * attn, dim=3)   # (N, H, W, C)
            if self.args.models.unet.use:
                if self.args.models.unet.double_channel:
                    fused_features = torch.cat([fused_features, fused_features], dim=-1)
                foreground = self.unet(fused_features.permute(0, 3, 1, 2), gamma=gamma, beta=beta).permute(0, 2, 3, 1).unsqueeze(-2)   # (N, H, W, 1, 3)
            elif self.args.models.fused_feature_mlp.use:
                fused_features = self.fused_feature_encoder(fused_features.reshape(N*H*W, -1))
                foreground = self.fused_feature_mlp(fused_features).reshape(N, H, W, 1, 3)
                fused_features = fused_features.reshape(N, H, W, -1)
            else:
                foreground = fused_features.unsqueeze(-2)
            rgb = foreground.squeeze(-2)
            topk_attn = attn  # For consistency with rest of code
            bkg_attn = torch.zeros(N, H, W, 1, 1, device=self.device)  # No background attention
        elif self.bkg_feats is not None and not self.args.geoms.points.use_bkg_points and not self.args.models.attn.append_bkg_points:
            bkg_seq_len = self.bkg_feats.shape[0]
            if self.bkg_token is None:
                scores = torch.cat([scores, self.bkg_score.expand(N, H, W, bkg_seq_len, -1)], dim=-2)
            # attn = F.softmax(scores, dim=3) # (N, H, W, num_pts+bkg_seq_len, 1)
            attn = self.get_attn_weights(scores)
            topk_attn = attn[..., :-1, :]
            bkg_attn = attn[..., -1:, :]
            if self.args.geoms.background.use_dumb_constant:
                bkg_attn = (bkg_attn * self.dumb_constant).clamp(0, 1)
            if self.args.models.normalize_topk_attn:
                topk_attn = topk_attn / (1 - bkg_attn + self.args.models.topk_eps)
            bkg_attn = self.bkg_attn_act((bkg_attn + self.args.models.bkg_attn_shift) * self.args.models.bkg_attn_scale)
            fused_features = torch.sum(embedv * topk_attn, dim=3)   # (N, H, W, C)

            if self.args.training.bkg_reg_mask_thresh > 0:
                bkg_mask = min_d2r[..., None] > self.args.training.bkg_reg_mask_thresh
                if bkg_mask.sum() > 0:
                    bkg_reg_loss = ((1 - bkg_attn[bkg_mask]) ** 2).mean()                

            if self.args.mask_bkg_rays_thresh > 0:
                fg_mask = min_d2r[..., None] < self.args.mask_bkg_rays_thresh
                bkg_attn_masked = torch.ones_like(bkg_attn, requires_grad=False)
                bkg_attn_masked[fg_mask] = bkg_attn[fg_mask]
            else:
                bkg_attn_masked = bkg_attn

            int_foreground = None
            if self.args.models.unet.use:
                if self.args.models.unet.double_channel:
                    fused_features = torch.cat([fused_features, fused_features], dim=-1)
                foreground = self.unet(fused_features.permute(0, 3, 1, 2), gamma=gamma, beta=beta).permute(0, 2, 3, 1).unsqueeze(-2)   # (N, H, W, 1, 3)
            elif self.args.models.fused_feature_mlp.use:
                fused_features = self.fused_feature_encoder(fused_features.reshape(N*H*W, -1))
                foreground = self.fused_feature_mlp(fused_features).reshape(N, H, W, 1, 3)
                fused_features = fused_features.reshape(N, H, W, -1)
            else:
                foreground = fused_features.unsqueeze(-2)

            # if self.args.models.normalize_topk_attn:
            #     if cur_step % 200 == 0:
            #         print(' topk_attn:', cur_step, topk_attn.shape, topk_attn.min().item(), topk_attn.max().item(), topk_attn.mean().item(), topk_attn.std().item())
            #         print(' bkg_attn:', cur_step, bkg_attn.shape, bkg_attn.min().item(), bkg_attn.max().item(), bkg_attn.mean().item(), bkg_attn.std().item())
            #     # bkg_attn = (bkg_attn * self.dumb_constant).clamp(0, 1)
            #     rgb = foreground * (1 - bkg_attn_masked) + bkg_color.expand(N, H, W, -1, -1) * bkg_attn_masked
            # else:
            #     rgb = foreground + bkg_color.expand(N, H, W, -1, -1) * bkg_attn_masked

            if cur_step % 200 == 0:
                print(' topk_attn:', cur_step, topk_attn.shape, topk_attn.min().item(), topk_attn.max().item(), topk_attn.mean().item(), topk_attn.std().item())
                print(' bkg_attn:', cur_step, bkg_attn.shape, bkg_attn.min().item(), bkg_attn.max().item(), bkg_attn.mean().item(), bkg_attn.std().item())

            # bkg_attn = (bkg_attn * self.dumb_constant).clamp(0, 1)
            if self.args.models.bkg_attn_deno_thresh > 0:
                bkg_deno = bkg_attn_masked.clone().detach()
                bkg_deno[bkg_deno < self.args.models.bkg_attn_deno_thresh] = 1
                bkg_attn_masked = bkg_attn_masked / bkg_deno
            if self.args.models.bkg_attn_sigmoid_scale > 0:
                bkg_attn_masked = torch.sigmoid(bkg_attn_masked * self.args.models.bkg_attn_sigmoid_scale)
            
            # Use cubemap textures for background if enabled and active
            cubemap_bkg_color = self.get_cubemap_background_color(rays_o, rays_d, cur_step)
            if cubemap_bkg_color is not None:
                # cubemap_bkg_color shape: (N, H, W, 3), expand to (N, H, W, 1, 3) for broadcasting
                rgb = foreground * (1 - bkg_attn_masked) + cubemap_bkg_color.unsqueeze(-2) * bkg_attn_masked
                if cur_step % 200 == 0:
                    print(' Using cubemap texture for background, shape:', cubemap_bkg_color.shape)
            else:
                rgb = foreground * (1 - bkg_attn_masked) + bkg_color.expand(N, H, W, -1, -1) * bkg_attn_masked
            rgb = rgb.squeeze(-2)
            int_rgb = None
            bkg_attn = bkg_attn.squeeze(-2)
        else:
            # attn = F.softmax(scores, dim=3)
            attn = self.get_attn_weights(scores)
            topk_attn = attn
            bkg_attn = attn

            if self.args.geoms.points.use_bkg_points and self.args.geoms.points.bkg_points_given_color:
                assert embedv.shape[-1] == 3, "bkg_points_given_color requires embedv to have 3 channels"
                selected_bkg_points_mask = self.bkg_points_mask[select_k_ind].expand(-1, -1, -1, -1, 3)
                embedv = torch.where(selected_bkg_points_mask == 1., embedv, bkg_color.expand(N, H, W, self.select_k, -1))
                
            if self.args.geoms.points.use_bkg_points and self.args.geoms.points.bkg_points_given_embedv:
                selected_bkg_points_mask = self.bkg_points_mask[select_k_ind].expand(-1, -1, -1, -1, 3)
                embedv = torch.where(selected_bkg_points_mask == 1., embedv, self.bkg_points_embedv.expand_as(embedv))
                
            if self.args.models.attn.append_bkg_points and getattr(self.args.models.attn, 'append_bkg_points_use_embedv', False) and self.append_bkg_points_embedv is not None:
                # Create a mask for the last element (appended background point) and use torch.where to avoid inplace operation
                append_bkg_mask = torch.zeros(embedv.shape[:-1], dtype=torch.bool, device=embedv.device)
                append_bkg_mask[..., -1] = True
                if self.append_bkg_points_embedv.ndim == 2:
                    if self.append_bkg_points_embedv.shape[0] == 4:
                        weights = torch.ones(4, device=self.device)
                        if bkg_color is not None:
                            weights[1:] = bkg_color.squeeze()
                        append_bkg_points_embedv = torch.sum(self.append_bkg_points_embedv * weights.unsqueeze(-1), dim=0)
                    else:
                        append_bkg_points_embedv = self.append_bkg_points_embedv.squeeze(0)
                else:
                    append_bkg_points_embedv = self.append_bkg_points_embedv
                append_bkg_embedv = append_bkg_points_embedv.expand(N, H, W, -1).unsqueeze(-2)
                embedv = torch.where(append_bkg_mask.unsqueeze(-1), append_bkg_embedv, embedv)
                
            if self.args.geoms.points.use_bkg_points and self.args.models.attn.append_bkg_points:
                topk_attn = attn[..., :-1, :]
                bkg_attn = attn[..., -1, :]     
                selected_bkg_points_mask = self.bkg_points_mask[select_k_ind].expand(-1, -1, -1, -1, 1)
                bkg_attn = torch.sum(topk_attn * (1 - selected_bkg_points_mask.float()), dim=3) + bkg_attn
            elif self.args.geoms.points.use_bkg_points:
                selected_bkg_points_mask = self.bkg_points_mask[select_k_ind].expand(-1, -1, -1, -1, 1)
                bkg_attn = torch.sum(attn * (1 - selected_bkg_points_mask.float()), dim=3)
            elif self.args.models.attn.append_bkg_points:
                topk_attn = attn[..., :-1, :]
                bkg_attn = attn[..., -1, :]

            fused_features = torch.sum(embedv * attn, dim=3)   # (N, H, W, C)

            if self.args.models.unet.use:
                if self.args.models.unet.double_channel:
                    fused_features = torch.cat([fused_features, fused_features], dim=-1)
                rgb = self.unet(fused_features.permute(0, 3, 1, 2), gamma=gamma, beta=beta).permute(0, 2, 3, 1)   # (N, H, W, 3)
            elif self.args.models.fused_feature_mlp.use:
                fused_features = self.fused_feature_encoder(fused_features.reshape(N*H*W, -1))
                rgb = self.fused_feature_mlp(fused_features).reshape(N, H, W, 3)
                fused_features = fused_features.reshape(N, H, W, -1)
            else:
                rgb = fused_features
                
            # Use cubemap textures for background if enabled and active
            cubemap_bkg_color = self.get_cubemap_background_color(rays_o, rays_d, cur_step)
            effective_bkg_color = cubemap_bkg_color if cubemap_bkg_color is not None else bkg_color.expand(N, H, W, -1)
            if cur_step % 200 == 0 and cubemap_bkg_color is not None:
                print(' Using cubemap texture for background (alt path), shape:', cubemap_bkg_color.shape)
            
            if self.args.geoms.points.use_bkg_points and self.args.models.attn.append_bkg_points:
                if self.args.geoms.points.bkg_points_alpha_blend or self.args.models.attn.append_bkg_points_alpha_blend:
                    rgb = rgb * (1 - bkg_attn) + effective_bkg_color * bkg_attn
            else:
                if self.args.geoms.points.use_bkg_points and self.args.geoms.points.bkg_points_alpha_blend:
                    rgb = rgb * (1 - bkg_attn) + effective_bkg_color * bkg_attn
                    
                if self.args.models.attn.append_bkg_points and self.args.models.attn.append_bkg_points_alpha_blend:
                    rgb = rgb * (1 - bkg_attn) + effective_bkg_color * bkg_attn                

        if log and cur_step >= 0 and cur_step % 200 == 0:
            print(' scores af:', cur_step, scores.shape, scores.min().item(), scores.max().item(), scores.mean().item(), scores.std().item())
            print(' feat map:', cur_step, fused_features.shape, fused_features.min().item(), fused_features.max().item(), fused_features.mean().item(), fused_features.std().item())
            print(' predict rgb:', cur_step, rgb.shape, rgb.min().item(), rgb.max().item(), rgb.mean().item(), rgb.std().item())
            print(' bkg_color:', cur_step, bkg_color.tolist())
            if self.bkg_points_embedv is not None:
                print(' bkg_points_embedv:', cur_step, self.bkg_points_embedv.shape, self.bkg_points_embedv.min().item(), self.bkg_points_embedv.max().item())
            
        # topk_loss = self.get_topk_mlp_loss(cur_step)
        # alpha_attn_loss = self.get_alpha_attn_loss(attn)
        # points_influ_scores_norm = torch.linalg.vector_norm(self.points_influ_scores) / self.points_influ_scores.numel()
        # points_influ_scores_mean = torch.zeros(0, device=self.device).sum()
        # if (self.points_influ_scores > 0).sum() > 0:
        #     points_influ_scores_mean = (self.points_influ_scores[self.points_influ_scores > 0].mean() - 0.5) ** 2
        
        # Compute all regularizer losses with scheduler support
        point_on_ray_loss, point_on_ray_loss_weight = self.compute_scheduled_loss(
            'point_on_ray_loss', cur_step,
            lambda: self.get_point_on_ray_loss(points, rays_o, rays_d, select_k_ind, attn, topk_attn, bkg_attn, mask))

        point_to_sp_loss, point_to_sp_loss_weight = self.compute_scheduled_loss(
            'point_to_sp_loss', cur_step,
            lambda: self.get_point_to_sp_loss(points, rays_o, rays_d, select_k_ind, attn, gt_mask=mask, step=cur_step))

        def compute_if_score_var():
            scores = self.points_influ_scores[self.points_influ_scores > 1.01e-5]
            return scores.var(dim=0).mean()
        if_score_var_loss, if_score_var_loss_weight = self.compute_scheduled_loss(
            'if_score_var_loss', cur_step, compute_if_score_var,
            extra_condition=self.points_influ_scores is not None)

        alpha_loss, alpha_loss_weight = self.compute_scheduled_loss(
            'alpha_loss', cur_step,
            lambda: torch.nn.functional.mse_loss(bkg_attn.squeeze(), (1-mask).detach().squeeze()))

        gt_pcd_loss, gt_pcd_loss_weight = self.compute_scheduled_loss(
            'gt_pcd_loss', cur_step,
            lambda: chamfer_distance(self.gt_points[None, ...], self.points[None, ...])[0].mean(),
            extra_condition=self.pruned_points and self.gt_points is not None)

        gt_sp_loss, gt_sp_loss_weight = self.compute_scheduled_loss(
            'gt_sp_loss', cur_step,
            lambda: self.get_gt_sp_loss(points, rays_o, rays_d, select_k_ind, attn, surface_points, gt_mask=mask, step=cur_step),
            extra_condition=self.pruned_points and surface_points is not None)

        cubemap_tv_loss, cubemap_tv_loss_weight = self.compute_scheduled_loss(
            'cubemap_tv_loss', cur_step, self.get_cubemap_tv_loss)

        # Regularizers dict: values are (loss_tensor, loss_weight) tuples
        # Weight is computed by get_scheduled_weight and passed directly
        regularizers = {
            # "prop_reg_loss": prop_reg_loss,
            # "bkg_reg_loss": bkg_reg_loss,
            # "topk_loss": topk_loss,
            # "alpha_attn_loss": alpha_attn_loss,
            # "influ_scores_norm": points_influ_scores_norm,
            # "influ_scores_mean": points_influ_scores_mean,
            "point_on_ray_loss": (point_on_ray_loss, point_on_ray_loss_weight),
            "point_to_sp_loss": (point_to_sp_loss, point_to_sp_loss_weight),
            "if_score_var_loss": (if_score_var_loss, if_score_var_loss_weight),
            "alpha_loss": (alpha_loss, alpha_loss_weight),
            "gt_pcd_loss": (gt_pcd_loss, gt_pcd_loss_weight),
            "gt_sp_loss": (gt_sp_loss, gt_sp_loss_weight),
        }

        return rgb, int_rgb, grid, regularizers, hit_pred, attn, embedv, select_k_ind, bkg_attn


    # def intersection_points_loss(self, rays_o, rays_d, c2w, pix_coords, pixels, mask, rbf, deformed_points, log_dir, og_intx_points, cur_step=-1):
    #     N, H, W, _ = rays_d.shape
    #     threshold = 0.5
    #     mask_intx_points = True
    #     use_deformed_loss = False
    #     save_pcds = True
    #     single_directional = True
        
    #     if self.args.geoms.points.select_k_rnd:
    #         self.select_k = random.randint(self.args.geoms.points.select_k, self.args.geoms.points.select_k_max)
            
    #     original_points = self.points
    #     if self.args.geoms.points.add_noise:
    #         original_points = self.add_gaussian_noise_to_points(original_points, cur_step)
    #         deformed_points = self.add_gaussian_noise_to_points(deformed_points, cur_step)

    #     original_points = self.get_points(original_points, cur_step=cur_step)
    #     deformed_points = self.get_points(deformed_points, cur_step=cur_step)
            
    #     original_points_2d, original_points_cam = fused_projection(original_points, c2w, self.fx, self.fy, self.cx, self.cy, self.H, self.W)
    #     deformed_points_2d, deformed_points_cam = fused_projection(deformed_points, c2w, self.fx, self.fy, self.cx, self.cy, self.H, self.W)
    #     original_z = original_points_cam[..., 2]
    #     deformed_z = deformed_points_cam[..., 2]

    #     original_intersection_points, original_attn, original_selected_points, original_select_k_ind = self.get_surface_points_from_rays(original_points, rays_o, rays_d, pix_coords, original_points_2d, original_z, cur_step, fuse_type="position")
    #     deformed_intersection_points, deformed_attn, deformed_selected_points, deformed_select_k_ind = self.get_surface_points_from_rays(deformed_points, rays_o, rays_d, pix_coords, deformed_points_2d, deformed_z, cur_step, fuse_type="position")
        
    #     if self.bkg_feats is not None and not self.args.geoms.points.use_bkg_points and not self.args.models.attn.append_bkg_points:
    #         deformed_topk_attn = deformed_attn[..., :-1, :]
    #         deformed_bkg_attn = deformed_attn[..., -1:, :]
    #         original_topk_attn = original_attn[..., :-1, :]
    #         original_bkg_attn = original_attn[..., -1:, :]
    #         if self.args.geoms.background.use_dumb_constant:
    #             original_bkg_attn = (original_bkg_attn * self.dumb_constant).clamp(0, 1)
    #             deformed_bkg_attn = (deformed_bkg_attn * self.dumb_constant).clamp(0, 1)
    #         if self.args.models.normalize_topk_attn:
    #             original_topk_attn = original_topk_attn / (1 - original_bkg_attn + self.args.models.topk_eps)
    #             deformed_topk_attn = deformed_topk_attn / (1 - deformed_bkg_attn + self.args.models.topk_eps)
    #         if self.args.models.normalize_topk_attn:
    #             deformed_topk_attn = deformed_topk_attn / (1 - deformed_bkg_attn + self.args.models.topk_eps)
    #         original_deformed_intx_points = self.get_surface_points(deformed_select_k_ind, original_points, rays_o, rays_d, deformed_attn, fuse_type="position")
    #     else:
    #         original_deformed_intx_points = self.get_surface_points(deformed_select_k_ind, original_points, rays_o, rays_d, deformed_attn, fuse_type="position")

    #     og_loss = torch.nn.functional.mse_loss(original_deformed_intx_points, og_intx_points)

    #     if mask_intx_points:
    #         original_deformed_intx_points = original_deformed_intx_points[deformed_bkg_attn.reshape(N, H, W) < threshold]

    #         original_intersection_points = original_intersection_points[original_bkg_attn.reshape(N, H, W) < threshold]
    #         deformed_intersection_points = deformed_intersection_points[deformed_bkg_attn.reshape(N, H, W) < threshold]

    #     # TODO: could weight points by bkg_attn
    #     # loss = chamfer_distance(original_intersection_points.reshape(1, -1, 3).detach(), original_deformed_intx_points.reshape(1, -1, 3), batch_reduction="mean", point_reduction="mean")[0].mean()
    #     loss = chamfer_distance(original_deformed_intx_points.reshape(1, -1, 3), original_intersection_points.reshape(1, -1, 3).detach(), 
    #                                 single_directional=single_directional, batch_reduction="mean", point_reduction="mean")[0].mean()

    #     if save_pcds and cur_step % 200 == 0:
    #         save_points(os.path.join(log_dir, f"original_intersection_points_{cur_step:04d}.ply"), original_intersection_points.detach().cpu().numpy() / self.coord_scale)
    #         save_points(os.path.join(log_dir, f"deformed_intersection_points_{cur_step:04d}.ply"), deformed_intersection_points.detach().cpu().numpy() / self.coord_scale)
    #         save_points(os.path.join(log_dir, f"original_deformed_intx_points_{cur_step:04d}.ply"), original_deformed_intx_points.detach().cpu().numpy() / self.coord_scale)
        
    #     if use_deformed_loss:
    #         deformed_original_intx_points = rbf(original_intersection_points.reshape(-1, 3).detach().cpu().numpy() / self.coord_scale)
    #         deformed_original_intx_points = torch.from_numpy(deformed_original_intx_points * self.coord_scale).to(self.device).to(torch.float)
    #         loss2 = chamfer_distance(deformed_intersection_points.reshape(1, -1, 3), deformed_original_intx_points.reshape(1, -1, 3).detach(),
    #                                     single_directional=single_directional, batch_reduction="mean", point_reduction="mean")[0].mean()
    #         loss = (loss + loss2) / 2  
    #         if save_pcds and cur_step % 200 == 0:
    #             save_points(os.path.join(log_dir, f"deformed_original_intx_points_{cur_step:04d}.ply"), deformed_original_intx_points.detach().cpu().numpy() / self.coord_scale)

    #     if cur_step % 200 == 0:
    #         print(f"loss: {loss.item()}, og_loss: {og_loss.item()}")

    #     return loss + og_loss
    
    
    def get_point_to_sp_loss(self, points, rays_o, rays_d, select_k_ind, attn, gt_mask=None, step=-1):  # detachattn = self.get_attn_weights(scores)
        use_bkg_points = False
        if self.args.geoms.no_additional_bkg:
            # No special background treatment - use all attention weights directly
            topk_attn = attn
            bkg_attn = torch.zeros_like(attn[..., :1, :])
            use_bkg_points = True  # Skip bkg_attn masking
        elif self.bkg_feats is not None and not self.args.geoms.points.use_bkg_points and not self.args.models.attn.append_bkg_points:
            topk_attn = attn[..., :-1, :]
            bkg_attn = attn[..., -1:, :]
            if self.args.models.point_to_sp_loss_norm_topk_attn:
                topk_attn = topk_attn / (1 - bkg_attn + self.eps)
        elif self.args.geoms.points.use_bkg_points and self.args.models.attn.append_bkg_points:
            topk_attn = attn[..., :-1, :]
            bkg_attn = attn[..., -1, :]
            selected_bkg_points_mask = self.bkg_points_mask[select_k_ind].expand(-1, -1, -1, -1, 1)
            bkg_attn = torch.sum(topk_attn * (1 - selected_bkg_points_mask.float()), dim=3) + bkg_attn
            bkg_attn = bkg_attn.unsqueeze(-1)
        elif self.args.models.attn.append_bkg_points:
            topk_attn = attn[..., :-1, :]
            bkg_attn = attn[..., -1:, :]
        else:
            use_bkg_points = True
            topk_attn = attn
            bkg_attn = attn
        selected_points = points[select_k_ind] # (N, H, W, K, C)
        if self.args.models.point_to_sp_loss_detach_attn:
            topk_attn = topk_attn.detach()
        sp = self.get_surface_points(select_k_ind, points, rays_o, rays_d, attn, self.args.models.point_to_sp_loss_fuse_type).unsqueeze(-2)
        if self.args.texture.divide_by_coord_scale:
            sp = sp * self.coord_scale
        if self.args.models.point_to_sp_loss_detach_sp:
            sp = sp.detach()
        distance_to_sp = torch.norm(selected_points - sp, dim=-1)   # (N, H, W, K, 1)
        if step >= 0 and step % 200 == 0:
            print(' distance_to_sp:', step, distance_to_sp.shape, distance_to_sp.min().item(), distance_to_sp.max().item(), distance_to_sp.mean().item(), distance_to_sp.std().item())
        if self.args.models.point_to_sp_loss_mask_thresh > 0:
            if self.args.models.point_to_sp_loss_use_gt_mask or use_bkg_points:
                mask = gt_mask.squeeze(-1) > 0.5
            else:
                mask = bkg_attn.squeeze(-1).squeeze(-1) < self.args.models.point_to_sp_loss_mask_thresh
            if self.args.models.point_to_sp_loss_erose_mask:
                kernel_size = 5
                # Use PyTorch erosion function that supports batch operations and stays on GPU
                mask = self.erode_mask(mask, kernel_size=kernel_size, iterations=self.args.models.point_to_sp_loss_erose_mask_iters)
            distance_to_sp = distance_to_sp[mask.detach()]
        if self.args.models.point_to_sp_loss_distance_thresh > 0 and not use_bkg_points:
            distance_to_sp = distance_to_sp[distance_to_sp < self.args.models.point_to_sp_loss_distance_thresh]
        return distance_to_sp.mean()
    
    
    def get_gt_sp_loss(self, points, rays_o, rays_d, select_k_ind, attn, gt_surface_points, gt_mask=None, step=-1):  # detachattn = self.get_attn_weights(scores)
        surface_points = self.get_surface_points(select_k_ind, points, rays_o, rays_d, attn, 'position-sphere').unsqueeze(-2)
        if self.args.texture.divide_by_coord_scale:
            gt_surface_points = gt_surface_points / self.coord_scale
        distance_to_sp = torch.norm(surface_points.squeeze(-2) - gt_surface_points, dim=-1)
        
        # Apply boundary mask filtering if enabled
        if self.args.training.gt_sp_loss_exclude_boundary and gt_mask is not None:
            boundary_width = self.args.training.gt_sp_loss_boundary_width
            # Create boundary mask: 0 at boundaries, 1 elsewhere
            boundary_mask = create_boundary_mask(gt_mask.squeeze(-1), boundary_width=boundary_width)
            # Apply mask: only compute loss on non-boundary pixels
            boundary_mask = boundary_mask.view_as(distance_to_sp)
            if step >= 0 and step % 200 == 0:
                num_boundary_pixels = (boundary_mask < 0.5).sum().item()
                num_total_pixels = boundary_mask.numel()
                print(' gt_sp_loss boundary mask: {}/{} pixels excluded (width={})'.format(
                    num_boundary_pixels, num_total_pixels, boundary_width))
            # Mask out boundary pixels
            distance_to_sp = distance_to_sp[boundary_mask > 0.5]
        
        if step >= 0 and step % 200 == 0:
            print(' distance_to_sp:', step, distance_to_sp.shape, distance_to_sp.min().item(), distance_to_sp.max().item(), distance_to_sp.mean().item(), distance_to_sp.std().item())
        return distance_to_sp.mean()
    
    
    def get_point_on_ray_loss(self, points, rays_o, rays_d, select_k_ind, attn, topk_attn, bkg_attn, gt_mask):
        use_bkg_points = False
        if self.args.geoms.no_additional_bkg:
            # No special background treatment - use all attention weights directly
            topk_attn = attn
            bkg_attn = torch.zeros_like(attn[..., :1, :])
            use_bkg_points = True  # Skip bkg_attn masking
        elif self.bkg_feats is not None and not self.args.geoms.points.use_bkg_points and not self.args.models.attn.append_bkg_points:
            if self.args.models.point_on_ray_loss_norm_topk_attn:
                topk_attn = topk_attn / (1 - bkg_attn + self.eps)
        elif self.args.models.attn.append_bkg_points:
            topk_attn = attn[..., :-1, :]
            bkg_attn = attn[..., -1:, :]
        else:
            use_bkg_points = True
            topk_attn = attn
            bkg_attn = attn
        selected_points = points[select_k_ind]
        if self.args.models.point_on_ray_loss_detach_points:
            selected_points = selected_points.detach()
        if self.args.models.point_on_ray_loss_type == "distance":
            metric = torch.sum(self.proximity_attn.d2r * topk_attn, dim=3)   # (N, H, W, 1)
        elif self.args.models.point_on_ray_loss_type == "distance2":
            surface_points = torch.sum(selected_points * topk_attn, dim=3, keepdim=True)
            pd, d2r, vec_pd, vec_d2r = self.proximity_attn.get_features(rays_o, rays_d, surface_points)
            metric = d2r.squeeze(-2)
        elif self.args.models.point_on_ray_loss_type == "distance3":
            N, H, W, _ = rays_d.shape
            if rays_o.dim() == 2:
                rays_o_hw = rays_o.reshape(N, 1, 1, 3).expand(-1, H, W, -1)
            elif rays_o.dim() == 4:
                rays_o_hw = rays_o
            else:
                raise ValueError(f"Unsupported rays_o ndim={rays_o.dim()}, expected 2 or 4")
            factor = (1 - (bkg_attn * self.dumb_constant).clamp(0, 1)) / (1 - bkg_attn + self.eps)
            bkg_attn = (bkg_attn * self.dumb_constant).clamp(0, 1)
            topk_attn = attn[..., :-1, :] * factor
            outer_points = rays_o_hw.unsqueeze(-2) + normalize_vector(rays_d).reshape(N, H, W, 1, 3) * 40
            surface_points = torch.sum(selected_points * topk_attn, dim=3, keepdim=True) + outer_points * bkg_attn
            pd, d2r, vec_pd, vec_d2r = self.proximity_attn.get_features(rays_o, rays_d, surface_points)
            metric = d2r.squeeze(-2)
        elif self.args.models.point_on_ray_loss_type == "angle":
            surface_points = torch.sum(selected_points * topk_attn, dim=3)
            if rays_o.dim() == 2:
                rays_o_hw = rays_o.reshape(rays_o.shape[0], 1, 1, 3)
            elif rays_o.dim() == 4:
                rays_o_hw = rays_o
            else:
                raise ValueError(f"Unsupported rays_o ndim={rays_o.dim()}, expected 2 or 4")
            rays = normalize_vector(surface_points - rays_o_hw)
            metric = torch.sum(rays * normalize_vector(rays_d), dim=-1, keepdim=True)   # (N, H, W, 1)
            metric = 1 - metric.clamp(0, 1)
        elif self.args.models.point_on_ray_loss_type == "vector":
            metric = torch.sum(self.proximity_attn.vec_d2r * topk_attn, dim=3)  # (N, H, W, 3)
        elif self.args.models.point_on_ray_loss_type == "vector2":
            surface_points = torch.sum(selected_points * topk_attn, dim=3, keepdim=True)
            pd, d2r, vec_pd, vec_d2r = self.proximity_attn.get_features(rays_o, rays_d, surface_points)
            metric = vec_d2r.squeeze(-2)
        else:
            raise ValueError("Unknown point on ray loss type: {}".format(self.args.models.point_on_ray_loss_type))
        if self.args.models.point_on_ray_loss_mask_thresh > 0:
            if self.args.models.point_on_ray_loss_use_gt_mask or use_bkg_points:
                mask = gt_mask.squeeze(-1) > 0.5
            else:
                mask = bkg_attn.squeeze(-1).squeeze(-1) < self.args.models.point_on_ray_loss_mask_thresh
            if self.args.models.point_on_ray_loss_erose_mask:
                kernel_size = 5
                # Use PyTorch erosion function that supports batch operations and stays on GPU
                mask = self.erode_mask(mask, kernel_size=kernel_size, iterations=self.args.models.point_on_ray_loss_erose_mask_iters)
            metric = metric[mask.detach()]
        return metric.mean()


    def get_texture_v(self, step, rays_o, rays_d, attn, selected_points, surface_points, nuvo_texture_map, selected_point_features=None, select_k_ind=None, z=None):
        if self.args.texture.v_type in [1]:
            pd, d2r, vec_pd, vec_d2r = self.proximity_attn.get_features(rays_o, rays_d, selected_points)
            v = self.proximity_attn.get_features_v(self.args.texture.v_type, rays_o, rays_d, selected_points, select_k_ind, vec_pd, vec_d2r, pd, d2r, z)
            N, H, W, K, D = v.shape
            num_pts = N * H * W * K
            v = self.texture_v_encoder(v.reshape(num_pts, D)) # (H*W*K, D_v)
            if selected_point_features is not None:
                v = torch.cat([v, selected_point_features.reshape(num_pts, -1)], dim=-1)
            if self.args.texture.network_v:
                v = self.texture_v_network(v)
            v = v.reshape(N, H, W, K, -1)
            topk_attn = attn[..., :-1, :]
            bkg_attn = attn[..., -1:, :]
            if self.args.models.normalize_topk_attn:
                topk_attn = topk_attn / (1 - bkg_attn + self.args.models.topk_eps)
            v_texture_map = torch.sum(v * topk_attn, dim=-2)
            if self.args.texture.v_detach:
                v_texture_map = v_texture_map.detach()
            nuvo_texture_map = torch.cat([v_texture_map, nuvo_texture_map], dim=-1)
        elif self.args.texture.v_type in [99]:
            N, H, W, _ = rays_d.shape
            nuvo_texture_map = self.texture_v_network(nuvo_texture_map.reshape(-1, nuvo_texture_map.shape[-1])).reshape(N, H, W, -1)
        else:
            v = self.proximity_attn.get_features_v(self.args.texture.v_type, rays_o, rays_d, surface_points, select_k_ind, None, None, None, None, z)
            N, H, W, D = v.shape
            num_pts = N * H * W
            if self.args.texture.v_detach:
                v = v.detach()
            v = self.texture_v_encoder(v.reshape(num_pts, D)) # (H*W*K, D_v)
            if selected_point_features is not None:
                v = torch.cat([v, selected_point_features.reshape(num_pts, -1)], dim=-1)
            if step >= 0:
                cur_v_weight = self.texture_v_scheduler(step)
                v = v * cur_v_weight
                if step % 200 == 0:
                    print("v weight:", step, cur_v_weight)
            if self.args.texture.fuse_v_with_nuvo_texture_map:
                v = torch.cat([nuvo_texture_map.reshape(num_pts, -1), v], dim=-1)
            if self.args.texture.network_v:
                v = self.texture_v_network(v)
            if self.args.texture.fuse_v_with_nuvo_texture_map:
                nuvo_texture_map = v.reshape(N, H, W, -1).float()
            else:
                nuvo_texture_map = torch.cat([nuvo_texture_map, v.reshape(N, H, W, -1)], dim=-1)
        return nuvo_texture_map
    
    
    def get_fused_chart_probs(self, select_k_ind, pcd_chart_probs, pcd_chart_logits, attn):
        selected_chart_probs = pcd_chart_probs[select_k_ind]
        if self.args.texture.use_bkg_score and not self.args.geoms.points.use_bkg_points:
            topk_attn = attn[..., :-1, :]
            bkg_attn = attn[..., -1:, :]
            if self.args.models.normalize_topk_attn:
                topk_attn = topk_attn / (1 - bkg_attn + self.args.models.topk_eps)
        else:
            topk_attn = attn
        if self.args.texture.fuse_probs_before_softmax:
            selected_chart_logits = pcd_chart_logits[select_k_ind]
            fused_chart_logits = torch.sum(selected_chart_logits * topk_attn, dim=-2)
            fused_chart_probs = F.softmax(fused_chart_logits, dim=-1)
        else:
            fused_chart_probs = torch.sum(selected_chart_probs * topk_attn, dim=-2)
        return fused_chart_probs, selected_chart_probs
    
    
    def get_fused_uvs(self, select_k_ind, pcd_pred_uvs, attn, selected_chart_probs, masks=None):
        selected_pred_uvs = [pcd_pred_uv[select_k_ind] for pcd_pred_uv in pcd_pred_uvs]
        if self.args.texture.use_bkg_score and not self.args.geoms.points.use_bkg_points:
            topk_attn = attn[..., :-1, :]
            bkg_attn = attn[..., -1:, :]
            if self.args.models.normalize_topk_attn:
                topk_attn = topk_attn / (1 - bkg_attn + self.args.models.topk_eps)
        else:
            topk_attn = attn
        fused_pred_uvs = []
        for chart_idx in range(len(selected_pred_uvs)):
            if masks is not None:
                mask = masks[chart_idx]
                if mask.sum() > 0:
                    sum_pred_uvs = selected_pred_uvs[chart_idx] * topk_attn
                    sum_pred_uvs = sum_pred_uvs.reshape(-1, self.select_k, 2)[mask]
                    if self.args.texture.fuse_uv_with_chart_probs:
                        masked_selected_chart_probs = selected_chart_probs.reshape(-1, self.select_k, len(pcd_pred_uvs))[mask]
                        fused_pred_uv = torch.sum(sum_pred_uvs * masked_selected_chart_probs[..., chart_idx:chart_idx+1], dim=-2)
                    else:
                        fused_pred_uv = torch.sum(sum_pred_uvs, dim=-2)
                else:
                    fused_pred_uv = torch.zeros((0, 2), dtype=torch.float32, device=self.device)
            else:
                if self.args.texture.fuse_uv_with_chart_probs:
                    fused_pred_uv = torch.sum(selected_pred_uvs[chart_idx] * topk_attn * selected_chart_probs[..., chart_idx:chart_idx+1], dim=-2)
                else:
                    fused_pred_uv = torch.sum(selected_pred_uvs[chart_idx] * topk_attn, dim=-2)
            fused_pred_uvs.append(fused_pred_uv)
        return fused_pred_uvs
    
    
    def get_surface_points(self, select_k_ind, points, rays_o, rays_d, attn, fuse_type):
        if self.args.geoms.no_additional_bkg:
            # No special background treatment - use all attention weights directly
            topk_attn = attn
        elif self.bkg_feats is not None and not self.args.geoms.points.use_bkg_points and not self.args.models.attn.append_bkg_points:
            topk_attn = attn[..., :-1, :]
            bkg_attn = attn[..., -1:, :]
            if self.args.geoms.background.use_dumb_constant:
                bkg_attn = (bkg_attn * self.dumb_constant).clamp(0, 1)
            if self.args.models.normalize_topk_attn:
                topk_attn = topk_attn / (1 - bkg_attn + self.args.models.topk_eps)
        elif self.args.geoms.points.use_bkg_points and self.args.models.attn.append_bkg_points:
            topk_attn = attn[..., :-1, :]
            bkg_attn = attn[..., -1, :]
            selected_bkg_points_mask = self.bkg_points_mask[select_k_ind].expand(-1, -1, -1, -1, 1)
            bkg_attn = torch.sum(topk_attn * (1 - selected_bkg_points_mask.float()), dim=3) + bkg_attn
            bkg_attn = bkg_attn.unsqueeze(-1)
        elif self.args.geoms.points.use_bkg_points:
            topk_attn = attn
            bkg_attn = attn
            selected_bkg_points_mask = self.bkg_points_mask[select_k_ind].expand(-1, -1, -1, -1, 1)
            bkg_attn = torch.sum(attn * (1 - selected_bkg_points_mask.float()), dim=3)
            bkg_attn = bkg_attn.unsqueeze(-1)
        elif self.args.models.attn.append_bkg_points:
            topk_attn = attn[..., :-1, :]
            bkg_attn = attn[..., -1:, :]
            bkg_attn = attn[..., -1:, :]
        else:
            topk_attn = attn
        if rays_o.ndim == 2:
            rays_o = rays_o.unsqueeze(1).unsqueeze(1)
        if fuse_type == "position":
            surface_points = torch.sum(points[select_k_ind] * topk_attn, dim=3)   # (N, H, W, C)
        elif fuse_type == "position-sphere":
            if self.args.geoms.no_additional_bkg:
                # No special background treatment - just use position
                surface_points = torch.sum(points[select_k_ind] * topk_attn, dim=3)   # (N, H, W, C)
            elif self.args.geoms.points.use_bkg_points and self.args.models.attn.append_bkg_points:
                surface_points = torch.sum(points[select_k_ind] * topk_attn, dim=3) + torch.sum(self.proximity_attn.sphere_intersection * bkg_attn, dim=3)
            elif self.args.geoms.points.use_bkg_points:
                surface_points = torch.sum(points[select_k_ind] * topk_attn, dim=3)   # (N, H, W, C)
            elif self.args.models.attn.append_bkg_points:
                surface_points = torch.sum(points[select_k_ind] * topk_attn, dim=3) + torch.sum(self.proximity_attn.sphere_intersection * bkg_attn, dim=3)
            else:
                # Use unified get_ray_sphere_intersection function
                bkg_points = self.get_bkg_sphere_intersection(rays_o, rays_d)
                surface_points = torch.sum(torch.cat([points[select_k_ind], bkg_points.unsqueeze(-2)], dim=3) * attn, dim=3)
        elif fuse_type == "position-sphere-projection":
            # First get positions the same way as "position-sphere"
            if self.args.geoms.no_additional_bkg:
                fused_points = torch.sum(points[select_k_ind] * topk_attn, dim=3)   # (N, H, W, C)
            elif self.args.geoms.points.use_bkg_points and self.args.models.attn.append_bkg_points:
                fused_points = torch.sum(points[select_k_ind] * topk_attn, dim=3) + torch.sum(self.proximity_attn.sphere_intersection * bkg_attn, dim=3)
            elif self.args.geoms.points.use_bkg_points:
                fused_points = torch.sum(points[select_k_ind] * topk_attn, dim=3)   # (N, H, W, C)
            elif self.args.models.attn.append_bkg_points:
                fused_points = torch.sum(points[select_k_ind] * topk_attn, dim=3) + torch.sum(self.proximity_attn.sphere_intersection * bkg_attn, dim=3)
            else:
                bkg_points = self.get_bkg_sphere_intersection(rays_o, rays_d)
                fused_points = torch.sum(torch.cat([points[select_k_ind], bkg_points.unsqueeze(-2)], dim=3) * attn, dim=3)
            # Then project the fused points onto the rays
            _, _, proj, _ = self.proximity_attn.get_features(rays_o, rays_d, fused_points.unsqueeze(-2))
            surface_points = (rays_o + proj.squeeze(-2)).squeeze(-2)
        elif fuse_type == "projection":
            fused_project_distances = torch.sum(self.proximity_attn.pd * topk_attn, dim=3)  # (N, H, W, 1)
            surface_points = (rays_o + normalize_vector(rays_d) * fused_project_distances)
        elif fuse_type == "position-projection":
            fused_points = torch.sum(points[select_k_ind] * topk_attn, dim=3)   # (N, H, W, 3)
            _, _, proj, _ = self.proximity_attn.get_features(rays_o, rays_d, fused_points.unsqueeze(-2))
            surface_points = (rays_o + proj.squeeze(-2)).squeeze(-2)
        else:
            raise ValueError("Unknown surface points fusion type: {}".format(fuse_type))
        
        if self.args.texture.divide_by_coord_scale:
            surface_points = surface_points / self.coord_scale
        return surface_points
    
    
    def get_surface_points_from_rays(self, points, sampled_rays_o, sampled_rays_d, sampled_pix_coords, sampled_points_2d, sampled_z, step, fuse_type="position"):
        B, H, W, _ = sampled_rays_d.shape
        select_k_ind, _ = self._get_topk_point_inds(sampled_rays_o, sampled_rays_d, points, sampled_pix_coords, sampled_points_2d, sampled_z, step)
        if self.args.rescale_bkg_points_for_attn:
            points_for_attn = points.clone()
            points_for_attn[self.bkg_points_mask == 0] /= self.args.rescale_bkg_points_for_attn_scale
        else:
            points_for_attn = points
        pc_feats = self.get_pc_feats()
        cur_points_influ_score = self.get_influ_scores()
        cur_points_scaler = self.get_scaler()
        append_bkg_points_feats = None
        if self.args.models.attn.append_bkg_points:
            # Check if we should use cubemap feature map for background point features
            if self.cubemap_use_feature_map:
                # Sample features from cubemap based on ray-sphere intersection directions
                cubemap_feats = self.get_cubemap_background_features(sampled_rays_o, sampled_rays_d, step)
                if cubemap_feats is not None:
                    # Shape: (N, H, W, feature_dim) -> (N, H, W, 1, feature_dim)
                    append_bkg_points_feats = cubemap_feats.unsqueeze(-2)
                elif self.append_bkg_points_feats is not None:
                    # Fallback to learned features if cubemap not active yet
                    append_bkg_points_feats = self.append_bkg_points_feats
            elif self.append_bkg_points_feats is not None:
                if self.append_bkg_points_feats.shape[0] == 4:
                    append_bkg_points_feats = torch.sum(self.append_bkg_points_feats, dim=0, keepdim=True)
                else:
                    append_bkg_points_feats = self.append_bkg_points_feats
        selected_points = points_for_attn[select_k_ind]
        
        # Get cubemap depth offset for background surface adjustment
        bkg_depth_offset = self.get_cubemap_depth_offset(sampled_rays_o, sampled_rays_d, cur_step=step)
        
        scores = self.proximity_attn(sampled_rays_o, sampled_rays_d, points_for_attn, pc_feats, cur_points_influ_score, select_k_ind, sampled_z, 
                                        c2w=None, bkg_token=self.bkg_token, step=step, max_step=self.args.training.steps, evaluate=False, scores_only=True, 
                                        append_bkg_points_feats=append_bkg_points_feats, points_scaler=cur_points_scaler, bkg_points_mask=self.bkg_points_mask,
                                        bkg_depth_offset=bkg_depth_offset)
        scores = scores.reshape(B, H, W, -1, 1)  # (B, H, W, select_k, 1)
        selected_influ_scores = cur_points_influ_score[select_k_ind] if cur_points_influ_score is not None else None
        if selected_influ_scores is not None:
            if self.args.influ_scores_fuse_type == "multiply":
                if self.bkg_token is not None or self.args.models.attn.append_bkg_points:
                    selected_influ_scores = torch.cat([selected_influ_scores, torch.ones(B, H, W, 1, 1, device=self.device)], dim=-2)
                scores = scores * selected_influ_scores
            elif self.args.influ_scores_fuse_type == "add":
                if self.bkg_token is not None or self.args.models.attn.append_bkg_points:
                    selected_influ_scores = torch.cat([selected_influ_scores, torch.zeros(B, H, W, 1, 1, device=self.device)], dim=-2)
                scores = scores + selected_influ_scores
            else:
                raise ValueError("Unknown influence scores fusion type: {}".format(self.args.influ_scores_fuse_type))
        
        if self.args.geoms.no_additional_bkg:
            # No special background treatment
            attn = self.get_attn_weights(scores)
            sampled_surface_points = self.get_surface_points(select_k_ind, points, sampled_rays_o, sampled_rays_d, attn, fuse_type)
        elif self.bkg_feats is not None and not self.args.geoms.points.use_bkg_points and not self.args.models.attn.append_bkg_points:
            bkg_seq_len = self.bkg_feats.shape[0]
            if self.bkg_token is None:
                scores = torch.cat([scores, self.bkg_score.expand(B, H, W, bkg_seq_len, -1)], dim=-2)
            attn = self.get_attn_weights(scores)
            sampled_surface_points = self.get_surface_points(select_k_ind, points, sampled_rays_o, sampled_rays_d, attn, fuse_type)
        else:
            attn = self.get_attn_weights(scores)
            sampled_surface_points = self.get_surface_points(select_k_ind, points, sampled_rays_o, sampled_rays_d, attn, fuse_type)
        return sampled_surface_points, attn, selected_points, select_k_ind
    

    @torch.no_grad()
    def sample_rays_batch(self, points, start_idx, num_samples, dataset, reshape=True, skip=1):
        rays_o = dataset.rayo[::skip]
        rays_d = dataset.rayd[::skip]
        masks = dataset.masks[::skip]
        c2w = dataset.c2w[::skip]
        pix_coords = dataset.pix_coords[::skip]
        # print(rays_o.shape, rays_d.shape, masks.shape, masks.dtype, c2w.shape)
        N, H, W, _ = rays_d.shape
        
        grid_H = dataset.args.patches.height
        grid_W = dataset.args.patches.width
        
        if reshape:
            assert num_samples % (grid_H * grid_W) == 0, "num_samples must be divisible by (grid_H * grid_W), {} vs {}".format(num_samples, grid_H * grid_W)
        B = num_samples // (grid_H * grid_W)

        points_2d, points_cam = fused_projection(points, c2w, dataset.focal_x, dataset.focal_y, dataset.cx, dataset.cy, H, W)
        z = points_cam[..., 2]
        # print(points_2d.shape, points_cam.shape, z.shape)

        num_rays = N*H*W
        sampled_indices = list(range(num_rays))[start_idx:start_idx+num_samples]


        num_points = points.shape[0]
        sampled_rays_o = rays_o.reshape(-1, 3)[sampled_indices]
        sampled_rays_d = rays_d.reshape(-1, 3)[sampled_indices]
        sampled_pix_coords = pix_coords.reshape(-1, 2)[sampled_indices]
        sampled_points_2d = points_2d.reshape(-1, 2)[sampled_indices]
        sampled_z = z.reshape(-1)[sampled_indices]

        if reshape:
            sampled_rays_o = sampled_rays_o.reshape(B, grid_H, grid_W, -1)
            sampled_rays_d = sampled_rays_d.reshape(B, grid_H, grid_W, -1)
            sampled_pix_coords = sampled_pix_coords.reshape(B, grid_H, grid_W, -1)
            sampled_points_2d = sampled_points_2d.reshape(B, grid_H, grid_W, num_points, -1)
            sampled_z = sampled_z.reshape(B, grid_H, grid_W, num_points)

        return sampled_rays_o, sampled_rays_d, sampled_pix_coords, sampled_points_2d, sampled_z
    
    
    @staticmethod
    def erode_mask(mask, kernel_size=5, iterations=1):
        """
        PyTorch implementation of morphological erosion for batched binary masks.
        Equivalent to cv2.erode but works on GPU and supports batch operations.
        
        Args:
            mask: Boolean tensor of shape (N, H, W) or (H, W)
            kernel_size: Size of the erosion kernel (default: 5)
            iterations: Number of erosion iterations (default: 1)
        
        Returns:
            Eroded boolean mask with same shape as input
        """
        # Handle single image case
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Convert boolean to float for processing
        mask_float = mask.float()
        padding = kernel_size // 2
        
        # Apply erosion iterations
        # Morphological erosion: output = 1 if ALL pixels in kernel neighborhood are 1, else 0
        # This is equivalent to min pooling, which we achieve using: 1 - max_pool2d(1 - input)
        result = mask_float
        for _ in range(iterations):
            result = result.unsqueeze(1)  # (N, 1, H, W) for conv operations
            
            # Erosion: min pooling using max_pool2d trick
            result_inv = 1 - result
            result_inv_pooled = F.max_pool2d(result_inv, kernel_size=kernel_size, stride=1, padding=padding)
            result = 1 - result_inv_pooled
            
            result = result.squeeze(1)  # (N, H, W)
        
        # Convert back to boolean
        result = result > 0.5
        
        if squeeze_output:
            result = result.squeeze(0)
        
        return result
    

    def save(self, step, save_dir):
        torch.save({str(step): self.state_dict()},
                   os.path.join(save_dir, 'model.pth'))

        optimizers_state_dict = {}
        for name, optimizer in self.optimizers.items():
            if optimizer is not None:
                optimizers_state_dict[name] = optimizer.state_dict()
            else:
                optimizers_state_dict[name] = None
        torch.save(optimizers_state_dict, os.path.join(
            save_dir, 'optimizers.pth'))

        schedulers_state_dict = {}
        for name, scheduler in self.schedulers.items():
            if scheduler is not None:
                schedulers_state_dict[name] = scheduler.state_dict()
            else:
                schedulers_state_dict[name] = None
        torch.save(schedulers_state_dict, os.path.join(
            save_dir, 'schedulers.pth'))
        
        scaler_state_dict = self.scaler.state_dict()
        torch.save(scaler_state_dict, os.path.join(
            save_dir, 'scaler.pth'))


    def load(self, load_dir, load_optimizer=False):
        model_state_dict = torch.load(os.path.join(load_dir, 'model.pth'), weights_only=True)
        step = list(model_state_dict.keys())[0]
        self.load_my_state_dict(model_state_dict[step])
        step = int(step)

        self.init_optimizers(step)        

        if load_optimizer == True:
            optimizers_state_dict = torch.load(
                os.path.join(load_dir, 'optimizers.pth'), weights_only=True)
            for name, optimizer in self.optimizers.items():
                if optimizer is not None:
                    optimizer.load_state_dict(optimizers_state_dict[name])
                else:
                    assert optimizers_state_dict[name] is None

            schedulers_state_dict = torch.load(
                os.path.join(load_dir, 'schedulers.pth'), weights_only=True)
            for name, scheduler in self.schedulers.items():
                if scheduler is not None:
                    scheduler.load_state_dict(schedulers_state_dict[name])
                else:
                    assert schedulers_state_dict[name] is None

        if os.path.exists(os.path.join(load_dir, 'scaler.pth')):
            scaler_state_dict = torch.load(
                os.path.join(load_dir, 'scaler.pth'), weights_only=True)
            self.scaler.load_state_dict(scaler_state_dict)

        return step


    def load_my_state_dict(self, state_dict, exclude=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            print("Loading", name, param.shape)
            for exclude_key in self.args.training.exclude_keys:
                if exclude_key in name and exclude:
                    print("exclude", name)
                    break
            else:
                if name not in ['points', 'points_influ_scores', 'points_scaler', 'pc_feats', 'points_last_grad', 'points_acc_grad', 'points_acc_grad_norm', 'points_grad_cnt', 'points_density', 'bkg_points_pc_feats', 'bkg_points_pc_feats_mlp', 'bkg_points_pc_feats_encoder', 'bkg_points_influ_scores', 'bkg_points']:
                    if isinstance(param, nn.Parameter):
                        # backwards compatibility for serialized parameters
                        param = param.data
                    try:
                        own_state[name].copy_(param)
                    except:
                        print("Can't load", name)

        self.points = nn.Parameter(state_dict['points'].data, requires_grad=self.points.requires_grad)
        self.points_last_grad = nn.Parameter(state_dict['points_last_grad'].data, requires_grad=False)
        self.points_acc_grad = nn.Parameter(state_dict['points_acc_grad'].data, requires_grad=False)
        self.points_acc_grad_norm = nn.Parameter(state_dict['points_acc_grad_norm'].data, requires_grad=False)
        self.points_grad_cnt = nn.Parameter(state_dict['points_grad_cnt'].data, requires_grad=False)
        if 'points_density' in state_dict:
            self.points_density = nn.Parameter(state_dict['points_density'].data, requires_grad=False)
        else:
            # Backward compatibility: compute density if not in checkpoint
            self.points_density = nn.Parameter(torch.ones(state_dict['points'].shape[0], device=self.device), requires_grad=False)
            self.update_point_density()
            print("Computed points_density from scratch (not found in checkpoint)")
        if self.points_influ_scores is not None:
            self.points_influ_scores = nn.Parameter(state_dict['points_influ_scores'].data, requires_grad=self.points_influ_scores.requires_grad)
        if 'points_scaler' in state_dict:
            self.points_scaler = nn.Parameter(state_dict['points_scaler'].data, requires_grad=self.points_scaler.requires_grad)
        else:
            # Backward compatibility: initialize points_scaler to ones if not in state_dict
            scaler_init_val = getattr(self.args.geoms.points, 'scaler_init_val', 1.0)
            self.points_scaler = nn.Parameter(torch.ones(state_dict['points'].shape[0], 1, device=self.device) * scaler_init_val, requires_grad=getattr(self.args.geoms.points, 'scaler_learn', False))
            print("Initialized points_scaler from scratch (not found in checkpoint)")
        self.pc_feats = nn.Parameter(state_dict['pc_feats'].data, requires_grad=self.pc_feats.requires_grad)
        print("load pc_feats", self.pc_feats.shape, self.pc_feats.min(), self.pc_feats.max())
        if self.bkg_points_pc_feats is not None:
            self.bkg_points_pc_feats = nn.Parameter(state_dict['bkg_points_pc_feats'].data, requires_grad=self.bkg_points_pc_feats.requires_grad)
        if self.bkg_points_pc_feats_mlp is not None and 'bkg_points_pc_feats_mlp.params' in state_dict:
            self.bkg_points_pc_feats_mlp.params.data.copy_(state_dict['bkg_points_pc_feats_mlp.params'].data)
        if self.bkg_points_influ_scores is not None:
            self.bkg_points_influ_scores = nn.Parameter(state_dict['bkg_points_influ_scores'].data, requires_grad=self.bkg_points_influ_scores.requires_grad)
        if self.bkg_points is not None:
            self.bkg_points = nn.Parameter(state_dict['bkg_points'].data, requires_grad=self.bkg_points.requires_grad)
