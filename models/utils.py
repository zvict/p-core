import torch
from torch import nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import scipy
from scipy.spatial import KDTree
import numpy as np
import math
import tinycudann as tcnn
import typing as T
import open3d as o3d


def parallel_greedy_assignment(
    neighbor_indices: torch.Tensor,
    neighbor_distances: torch.Tensor
) -> torch.Tensor:
    """
    Assigns a single unique nearest neighbor to each sample point in a parallel, batched manner.

    Args:
        neighbor_indices (torch.Tensor): Tensor of shape (B, N, K) containing indices of M nearest
                                        neighbors for each of N sample points per ray.
                                        B=N_rays, N=K_samples, K=M_neighbors.
        neighbor_distances (torch.Tensor): Tensor of shape (B, N, K) with corresponding distances.

    Returns:
        torch.Tensor: A tensor of shape (B, N) with the final assigned unique neighbor
                      index for each sample point. Unassigned points will have -1.
    """
    # Get tensor shapes and device
    B, N, K = neighbor_indices.shape
    device = neighbor_indices.device

    # === Step 1: Create a flat list of all candidate pairings ===
    # Create an index to track which of the N sample points each candidate belongs to
    sample_indices = torch.arange(N, device=device).view(1, N, 1).expand(B, N, K)

    # Flatten the last two dimensions to get a list of candidates per ray
    # Shape of each becomes (B, N * K)
    flat_distances = neighbor_distances.reshape(B, -1)
    flat_neighbors = neighbor_indices.reshape(B, -1)
    flat_samples = sample_indices.reshape(B, -1)

    # === Step 2: Sort all candidates by distance in ascending order ===
    # The sort_indices will be used to reorder all three tensors consistently
    sorted_dist_indices = torch.argsort(flat_distances, dim=1)

    # Gather the tensors in the new sorted order
    # Each row now represents a candidate pairing (sample_id, neighbor_id)
    # sorted from best to worst distance.
    sorted_neighbors = torch.gather(flat_neighbors, 1, sorted_dist_indices)
    sorted_samples = torch.gather(flat_samples, 1, sorted_dist_indices)

    # === Step 3: First pass - Enforce unique neighbor_id ===
    # We find the first time each neighbor appears in the sorted list.
    # This ensures that the point cloud point is assigned to its closest sample.
    
    # A trick to find the first occurrence index in a batched way:
    # We sort the sorted_neighbors tensor itself, which groups identical IDs.
    # The first in each group is the one we want.
    unique_neighbors, unique_inverse, unique_counts = torch.unique(
        sorted_neighbors, dim=1, sorted=True, return_inverse=True, return_counts=True
    )
    
    # The perm tensor gives the original positions of the sorted unique values
    perm = torch.arange(unique_inverse.size(1), device=device)
    perm = unique_inverse.new_empty(unique_neighbors.size(1)).scatter_(0, unique_inverse[0], perm)
    
    # After this pass, each neighbor_id appears at most once
    pass1_neighbors = sorted_neighbors[:, perm]
    pass1_samples = sorted_samples[:, perm]

    # === Step 4: Second pass - Enforce unique sample_id ===
    # Now, from the valid pairs, we ensure each sample_id gets only its best
    # (i.e., first) assignment.
    unique_samples, unique_inverse, unique_counts = torch.unique(
        pass1_samples, dim=1, sorted=True, return_inverse=True, return_counts=True
    )
    perm = torch.arange(unique_inverse.size(1), device=device)
    perm = unique_inverse.new_empty(unique_samples.size(1)).scatter_(0, unique_inverse[0], perm)

    # These are our final, unique (sample_id, neighbor_id) pairs
    final_samples = pass1_samples[:, perm]
    final_neighbors = pass1_neighbors[:, perm]

    # === Step 5: Scatter results into the final output tensor ===
    # Create a placeholder tensor
    output = torch.full((B, N), -1, dtype=torch.long, device=device)
    
    # Place the final neighbor assignments into the correct sample slot
    output.scatter_(dim=1, index=final_samples, src=final_neighbors)
    
    return output


def load_points(file_path):
    """Load point cloud from a .xyz, .ply, .npy or .npz file."""
    if file_path.endswith('.xyz') or file_path.endswith('.txt'):
        return np.loadtxt(file_path, delimiter=' ', dtype=float)
    elif file_path.endswith('.ply'):
        import open3d as o3d
        points = o3d.io.read_point_cloud(file_path)
        return np.asarray(points.points)
    elif file_path.endswith('.npy'):
        return np.load(file_path)
    elif file_path.endswith('.npy') or file_path.endswith('.npz'):
        with np.load(file_path) as data:
            return data['points']
    else:
        raise ValueError(f"Unsupported file format: {file_path.split('.')[-1]}")


def save_points(file_path, points):
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    points = points.reshape(-1, 3)
    if file_path.endswith('.xyz'):
        np.savetxt(file_path, points, delimiter=' ')
    elif file_path.endswith('.ply'):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(file_path, pcd)
    else:
        raise ValueError(f"Unsupported file format: {file_path.split('.')[-1]}")


def inverse_sigmoid(x):
    return torch.log(x/(1-x))


def get_tcnn_init_weights(input_dims, output_dims, network_config, *, device):
    if network_config.n_hidden_layers > 0:
        modules = [nn.Linear(input_dims, network_config.n_neurons, bias=False), activation_func(network_config.activation)]
        for i in range(network_config.n_hidden_layers - 1):
            modules.append(nn.Linear(network_config.n_neurons, network_config.n_neurons, bias=False))
            modules.append(activation_func(network_config.activation))
        modules.append(nn.Linear(network_config.n_neurons, output_dims, bias=False))
        if network_config.output_activation != "None":
            modules.append(activation_func(network_config.output_activation))
    else:
        # modules = [nn.Linear(input_dims, output_dims, bias=False)]
        # if network_config.output_activation != "None":
        #     modules.append(activation_func(network_config.output_activation))
        raise NotImplementedError("No hidden layers in the network")

    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

    # tiny-cuda-nn exposes its packed parameter tensor on the device selected
    # for the network.  Build the equivalent torch layers there as well rather
    # than silently targeting CUDA device 0.
    model = nn.Sequential(*modules).to(device=device)
    model.apply(_init_weights)

    for n, p in model.named_parameters():
        print(n, p.shape)

    linear_layers = [m for m in model if isinstance(m, nn.Linear)]
    input_layer_weights = linear_layers[0].weight.data
    output_layer_weights = linear_layers[-1].weight.data
    if input_dims % 16 != 0:
        input_layer_weights = nn.functional.pad(input_layer_weights, (0, 16 - (input_dims % 16)), value=0)  # Pad with 1, check https://github.com/NVlabs/tiny-cuda-nn/issues/6
    if output_dims % 16 != 0:
        output_layer_weights = nn.functional.pad(output_layer_weights, (0, 0, 0, 16 - (output_dims % 16)), value=0)

    weights = [input_layer_weights] + [m.weight.data for m in linear_layers[1:-1]] + [output_layer_weights]
    for w in weights:
        print(w.shape)

    weights = torch.cat([w.flatten() for w in weights]).half()
    return weights


class LayerNorm(nn.Module):
    "Construct a layernorm module"

    def __init__(self, feat_dim, elementwise_affine=True, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weights = nn.Parameter(torch.ones(feat_dim))
            self.bias = nn.Parameter(torch.zeros(feat_dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        # std = x.std(-1, keepdim=True) # sqrt(0) gives nan gradientsm happens when the inputs are constant
        var = x.var(-1, keepdim=True)
        std = torch.sqrt(var + self.eps)
        if self.elementwise_affine:
            return self.weights * (x - mean) / std + self.bias
        else:
            return (x - mean) / std
    

class Encoding(nn.Module):
    def __init__(self, input_dim, config, use_tcnn_encoder=True):
        super(Encoding, self).__init__()
        self.config = config
        self.use_tcnn_encoder = use_tcnn_encoder
        if config.n_frequencies > 0:
            if use_tcnn_encoder:
                self.encoder = tcnn.Encoding(input_dim, config, dtype=torch.float32)
                self.n_output_dims = self.encoder.n_output_dims + input_dim if config.with_self else self.encoder.n_output_dims
            else:
                self.encoder = PoseEnc(mult_factor=config.pe_scale)
                n_output_dims = input_dim * 2 * config.n_frequencies
                self.n_output_dims = n_output_dims + input_dim if config.with_self else n_output_dims
        else:
            assert config.with_self
            self.n_output_dims = input_dim
        print("Encoding output dim: ", self.n_output_dims, input_dim, self.use_tcnn_encoder)

    def forward(self, x):
        if self.config.n_frequencies > 0:
            if self.use_tcnn_encoder:
                # tinycudann scale the elements by pi while NeRF does not, and this scaling is super important, 
                # without it the attention won't recover after the first-time pruning, worth further investigation
                code = self.encoder(x * self.config.pe_scale / torch.pi)
                if self.config.with_self:
                    if self.config.stop_gradient:
                        code = torch.cat([x, code.detach()], dim=-1)
                    else:
                        code = torch.cat([x, code], dim=-1)
            else:
                code = self.encoder(x, self.config.n_frequencies, without_self=not self.config.with_self, stop_gradient=self.config.stop_gradient)
        else:
            code = x
        return code


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def add_points_knn(coords, influ_scores, add_num, k, comb_type="mean", sample_type="random", sample_k=10, 
                   point_features=None, point_alphas=None, point_scalers=None, last_coord_grad=None, acc_coord_grad=None, 
                   acc_coord_grad_norm=None, grad_cnt=None, move_scale=2.0, hybrid_weight=0.5):
    """
    Add points to the point cloud by kNN
    """
    N = coords.shape[0]

    # Step 1: Determine where to add points
    if N <= add_num and "random" in comb_type:
        inds = np.random.choice(N, add_num, replace=True)
        query_coords = coords[inds, :]
    elif N <= add_num:
        query_coords = coords
        inds = list(range(N))
    else:
        if sample_type == "random":
            inds = np.random.choice(N, add_num, replace=False)
            query_coords = coords[inds, :]
        elif sample_type == "top-knn-std":
            assert k >= 2
            pc = KDTree(coords)
            nns_dists, nns_inds = pc.query(coords, k=sample_k)
            inds = np.argsort(nns_dists.std(axis=-1))[-add_num:]
            query_coords = coords[inds, :]
        elif sample_type == "top-knn-mean":
            assert k >= 2
            pc = KDTree(coords)
            nns_dists, nns_inds = pc.query(coords, k=sample_k)
            inds = np.argsort(nns_dists.mean(axis=-1))[-add_num:]
            query_coords = coords[inds, :]
        elif sample_type == "top-knn-max":
            assert k >= 2
            pc = KDTree(coords)
            nns_dists, nns_inds = pc.query(coords, k=sample_k)
            inds = np.argsort(nns_dists.max(axis=-1))[-add_num:]
            query_coords = coords[inds, :]
        elif sample_type == "top-knn-min":
            assert k >= 2
            pc = KDTree(coords)
            nns_dists, nns_inds = pc.query(coords, k=sample_k)
            inds = np.argsort(nns_dists.min(axis=-1))[-add_num:]
            query_coords = coords[inds, :]
        elif sample_type == "influ-scores-max":
            inds = np.argsort(influ_scores.squeeze())[-add_num:]
            query_coords = coords[inds, :]
        elif sample_type == "influ-scores-min":
            inds = np.argsort(influ_scores.squeeze())[:add_num]
            query_coords = coords[inds, :]
        elif sample_type == "last-coord-grad-max":
            inds = np.argsort(last_coord_grad.abs().sum(-1))[-add_num:]
            query_coords = coords[inds, :]
        elif sample_type == "acc-coord-grad-max":
            inds = np.argsort(acc_coord_grad.abs().sum(-1))[-add_num:]
            query_coords = coords[inds, :]
        elif sample_type == "acc-coord-grad-cnt-max":
            inds = np.argsort(acc_coord_grad.abs().sum(-1) / (grad_cnt + 1))[-add_num:]
            query_coords = coords[inds, :]
        elif sample_type == "acc-coord-grad-norm-max":  # the togo now
            inds = np.argsort(acc_coord_grad_norm)[-add_num:]
            query_coords = coords[inds, :]
        elif sample_type == "acc-coord-grad-norm-cnt-max":
            inds = np.argsort(acc_coord_grad_norm / (grad_cnt + 1))[-add_num:]
            query_coords = coords[inds, :]
        elif sample_type == "acc-coord-grad-norm-max-hybrid-top-knn-std":
            inds_a = np.argsort(acc_coord_grad_norm)
            ranks_a = np.zeros_like(inds_a)
            ranks_a[inds_a] = np.arange(len(inds_a))
            
            assert k >= 2
            pc = KDTree(coords)
            nns_dists, nns_inds = pc.query(coords, k=sample_k+1)
            nns_dists = nns_dists[:, 1:]
            inds_b = np.argsort(nns_dists.std(axis=-1))
            ranks_b = np.zeros_like(inds_b)
            ranks_b[inds_b] = np.arange(len(inds_b))

            ranks = hybrid_weight * ranks_a + (1 - hybrid_weight) * ranks_b
            inds = np.argsort(ranks)[-add_num:]
            query_coords = coords[inds, :]
        else:
            raise NotImplementedError

    # Step 2: Add points by kNN
    new_features = None
    new_alphas = None
    new_scalers = None
    if comb_type == "duplicate":
        noise = np.random.randn(3).astype(np.float32)
        noise = noise / np.linalg.norm(noise)
        noise *= k
        new_coords = (query_coords + noise)
        new_influ_scores = influ_scores[inds, :]
        if point_features is not None:
            new_features = point_features[inds, :]
        if point_alphas is not None:
            new_alphas = point_alphas[inds, :]
        if point_scalers is not None:
            new_scalers = point_scalers[inds, :]
    elif comb_type == "clone":
        new_coords = query_coords
        new_influ_scores = influ_scores[inds, :]
        if point_features is not None:
            new_features = point_features[inds, :]
        if point_alphas is not None:
            new_alphas = point_alphas[inds, :]
        if point_scalers is not None:
            new_scalers = point_scalers[inds, :]
    else:
        pc = KDTree(coords)
        nns_dists, nns_inds = pc.query(query_coords, k=k+1)
        nns_dists = nns_dists.astype(np.float32)
        nns_dists = nns_dists[:, 1:]
        nns_inds = nns_inds[:, 1:]
        if comb_type == "mean":
            new_coords = coords[nns_inds, :].mean(
                axis=-2)  # (Nq, k, 3) -> (Nq, 3)
            new_influ_scores = influ_scores[nns_inds, :].mean(axis=-2)
            if point_features is not None:
                new_features = point_features[nns_inds, :].mean(axis=-2)
            if point_alphas is not None:
                new_alphas = point_alphas[nns_inds, :].mean(axis=-2)
            if point_scalers is not None:
                new_scalers = point_scalers[nns_inds, :].mean(axis=-2)
        elif comb_type == "random":
            rnd_w = np.random.uniform(0, 1, (query_coords.shape[0], k)).astype(np.float32)
            rnd_w /= rnd_w.sum(axis=-1, keepdims=True)
            new_coords = (coords[nns_inds, :] * rnd_w.reshape(-1, k, 1)).sum(axis=-2)
            new_influ_scores = (influ_scores[nns_inds, :] * rnd_w.reshape(-1, k, 1)).sum(axis=-2)
            if point_features is not None:
                new_features = (point_features[nns_inds, :] * rnd_w.reshape(-1, k, 1)).sum(axis=-2)
            if point_alphas is not None:
                new_alphas = (point_alphas[nns_inds, :] * rnd_w.reshape(-1, k, 1)).sum(axis=-2)
            if point_scalers is not None:
                new_scalers = (point_scalers[nns_inds, :] * rnd_w.reshape(-1, k, 1)).sum(axis=-2)
        elif comb_type == "random-softmax":
            rnd_w = np.random.randn(query_coords.shape[0], k).astype(np.float32)
            rnd_w = scipy.special.softmax(rnd_w, axis=-1)
            new_coords = (coords[nns_inds, :] * rnd_w.reshape(-1, k, 1)).sum(axis=-2)
            new_influ_scores = (influ_scores[nns_inds, :] * rnd_w.reshape(-1, k, 1)).sum(axis=-2)
            if point_features is not None:
                new_features = (point_features[nns_inds, :] * rnd_w.reshape(-1, k, 1)).sum(axis=-2)
            if point_alphas is not None:
                new_alphas = (point_alphas[nns_inds, :] * rnd_w.reshape(-1, k, 1)).sum(axis=-2)
            if point_scalers is not None:
                new_scalers = (point_scalers[nns_inds, :] * rnd_w.reshape(-1, k, 1)).sum(axis=-2)
        elif comb_type == "weighted":
            new_coords = (coords[nns_inds, :] * (1 / (nns_dists + 1e-6)).reshape(-1, k, 1)).sum(axis=-2) / (1 / (nns_dists + 1e-6)).sum(axis=-1, keepdims=True)
            new_influ_scores = (influ_scores[nns_inds, :] * (1 / (nns_dists + 1e-6)).reshape(-1, k, 1)).sum(axis=-2) / (1 / (nns_dists + 1e-6)).sum(axis=-1, keepdims=True)
            if point_features is not None:
                new_features = (point_features[nns_inds, :] * (1 / (nns_dists + 1e-6)).reshape(-1, k, 1)).sum(axis=-2) / (1 / (nns_dists + 1e-6)).sum(axis=-1, keepdims=True)
            if point_alphas is not None:
                new_alphas = (point_alphas[nns_inds, :] * (1 / (nns_dists + 1e-6)).reshape(-1, k, 1)).sum(axis=-2) / (1 / (nns_dists + 1e-6)).sum(axis=-1, keepdims=True)
            if point_scalers is not None:
                new_scalers = (point_scalers[nns_inds, :] * (1 / (nns_dists + 1e-6)).reshape(-1, k, 1)).sum(axis=-2) / (1 / (nns_dists + 1e-6)).sum(axis=-1, keepdims=True)
        elif comb_type == "along-last-coord-grad":
            new_coords = query_coords + last_coord_grad[inds, :] * move_scale
            nns_dists, nns_inds = pc.query(new_coords, k=k)
            nns_dists = nns_dists.astype(np.float32)
            new_influ_scores = (influ_scores[nns_inds, :] * (1 / (nns_dists + 1e-6)).reshape(-1, k, 1)).sum(axis=-2) / (1 / (nns_dists + 1e-6)).sum(axis=-1, keepdims=True)
            if point_features is not None:
                new_features = (point_features[nns_inds, :] * (1 / (nns_dists + 1e-6)).reshape(-1, k, 1)).sum(axis=-2) / (1 / (nns_dists + 1e-6)).sum(axis=-1, keepdims=True)
            if point_alphas is not None:
                new_alphas = (point_alphas[nns_inds, :] * (1 / (nns_dists + 1e-6)).reshape(-1, k, 1)).sum(axis=-2) / (1 / (nns_dists + 1e-6)).sum(axis=-1, keepdims=True)
            if point_scalers is not None:
                new_scalers = (point_scalers[nns_inds, :] * (1 / (nns_dists + 1e-6)).reshape(-1, k, 1)).sum(axis=-2) / (1 / (nns_dists + 1e-6)).sum(axis=-1, keepdims=True)
        elif comb_type == "along-acc-coord-grad":
            new_coords = query_coords + acc_coord_grad[inds, :] * move_scale
            nns_dists, nns_inds = pc.query(new_coords, k=k)
            nns_dists = nns_dists.astype(np.float32)
            new_influ_scores = (influ_scores[nns_inds, :] * (1 / (nns_dists + 1e-6)).reshape(-1, k, 1)).sum(axis=-2) / (1 / (nns_dists + 1e-6)).sum(axis=-1, keepdims=True)
            if point_features is not None:
                new_features = (point_features[nns_inds, :] * (1 / (nns_dists + 1e-6)).reshape(-1, k, 1)).sum(axis=-2) / (1 / (nns_dists + 1e-6)).sum(axis=-1, keepdims=True)
            if point_alphas is not None:
                new_alphas = (point_alphas[nns_inds, :] * (1 / (nns_dists + 1e-6)).reshape(-1, k, 1)).sum(axis=-2) / (1 / (nns_dists + 1e-6)).sum(axis=-1, keepdims=True)
            if point_scalers is not None:
                new_scalers = (point_scalers[nns_inds, :] * (1 / (nns_dists + 1e-6)).reshape(-1, k, 1)).sum(axis=-2) / (1 / (nns_dists + 1e-6)).sum(axis=-1, keepdims=True)
        else:
            raise NotImplementedError
    return new_coords, len(new_coords), new_influ_scores, new_features, new_alphas, new_scalers, inds


def transform_points(coords, c2w, vector=False):
    if vector:  # Convert to homogeneous coordinates
        coords = torch.cat([coords, torch.zeros_like(coords[..., :1])], -1)
    else:
        coords = torch.cat([coords, torch.ones_like(coords[..., :1])], -1)

    if coords.ndim == 5:
        assert c2w.ndim == 2
        B, H, W, N, _ = coords.shape
        transformed_coords = torch.sum(
            coords.unsqueeze(-2) * c2w.reshape(1, 1, 1, 1, 4, 4), -1)    # [B, H, W, N, 3]
    elif coords.ndim == 4:
        assert c2w.ndim == 3
        N, H, W, _ = coords.shape
        transformed_coords = torch.sum(
            coords.unsqueeze(-2) * c2w.reshape(N, 1, 1, 4, 4), -1)  # [N, H, W, 4]
    elif coords.ndim == 3:
        assert c2w.ndim == 2
        H, W, _ = coords.shape
        transformed_coords = torch.sum(
            coords.unsqueeze(-2) * c2w.reshape(1, 1, 4, 4), -1)    # [H, W, 4]
    elif coords.ndim == 2:
        K, _ = coords.shape
        if c2w.ndim == 2:
            transformed_coords = torch.sum(coords.unsqueeze(-2) * c2w.reshape(1, 4, 4), -1)   # [K, 4]
        elif c2w.ndim == 3:
            transformed_coords = torch.sum(coords[None, :, None, :] * c2w.reshape(-1, 1, 4, 4), -1)
    else:
        raise ValueError('Wrong dimension of coords')
    return transformed_coords[..., :3]


def cam_to_world(coords, c2w, vector=True):
    """
        coords: [N, H, W, 3] or [H, W, 3] or [K, 3]
        c2w: [N, 4, 4] or [4, 4]
    """
    return transform_points(coords, c2w, vector)


def world_to_cam(coords, c2w, vector=True):
    """
        coords: [N, H, W, 3] or [H, W, 3] or [K, 3]
        c2w: [N, 4, 4] or [4, 4]
    """
    w2c = torch.inverse(c2w)
    return transform_points(coords, w2c, vector)


# def perspective_projection(coords_cam, fx, fy, cx, cy, H, W):
#     depth = -coords_cam[..., 2] + 1e-6
#     u = (coords_cam[..., 0] * fx / depth + cx)
#     v = -(coords_cam[..., 1] * fy / depth + cy)
#     coords_2d = torch.stack([u, v], -1)
#     return coords_2d


def perspective_projection(coords, fx, fy, cx, cy, H, W):
    rz = 1 / (coords[..., 2] + 1e-6)
    coords_2d_x = -(coords[..., 0] * fx * rz + cx) + W
    coords_2d_y = (coords[..., 1] * fy * rz + cy)
    coords_2d = torch.stack([coords_2d_x, coords_2d_y], -1)
    return coords_2d


def fused_projection(coords, c2w, fx, fy, cx, cy, H, W, vector=False):
    coords_cam = world_to_cam(coords, c2w, vector)
    # mask = coords_cam[..., 2] > 1e-6
    coords_2d = perspective_projection(coords_cam, fx, fy, cx, cy, H, W)
    return coords_2d, coords_cam


def activation_func(act_type='leakyrelu', neg_slope=0.2, inplace=True, num_channels=128, a=1., b=1., trainable=False):
    act_type = act_type.lower()
    if act_type == 'none':
        layer = nn.Identity()
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_channels)
    elif act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == '+1':
        layer = PlusOneActivation()
    elif act_type == 'relu+1':
        layer = nn.Sequential(nn.ReLU(inplace), PlusOneActivation())
    elif act_type == 'tanh':
        layer = nn.Tanh()
    elif act_type == 'shifted_tanh':
        layer = ShiftedTanh()
    elif act_type == 'sigmoid':
        layer = nn.Sigmoid()
    elif act_type == 'gelu':
        layer = nn.GELU()
    elif act_type == 'swish' or act_type == 'silu':
        layer = nn.SiLU(inplace)
    elif act_type == 'gaussian':
        layer = GaussianActivation(a, trainable)
    elif act_type == 'quadratic':
        layer = QuadraticActivation(a, trainable)
    elif act_type == 'multi-quadratic':
        layer = MultiQuadraticActivation(a, trainable)
    elif act_type == 'laplacian':
        layer = LaplacianActivation(a, trainable)
    elif act_type == 'super-gaussian':
        layer = SuperGaussianActivation(a, b, trainable)
    elif act_type == 'expsin':
        layer = ExpSinActivation(a, trainable)
    elif act_type == 'clamp':
        layer = Clamp(0, 1)
    elif act_type == 'sh_plus_half_min':
        layer = SHPlusHalfMin(0)
    elif 'sine' in act_type:
        layer = Sine(factor=a)
    elif 'softplus' in act_type:
        a, b, c = [float(i) for i in act_type.split('_')[1:]]
        # print('Softplus activation: a={:.2f}, b={:.2f}, c={:.2f}'.format(a, b, c))
        layer = SoftplusActivation(a, b, c)
    elif act_type == 'flipped_elu':
        layer = FlippedELU()
    else:
        raise NotImplementedError(
            'activation layer [{:s}] is not found'.format(act_type))
    return layer


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def posenc(x, L_embed, factor=2.0, without_self=False, mult_factor=1.0, stop_gradient=False):
    if without_self:
        rets = []
    else:
        rets = [x]
    for i in range(L_embed):
        for fn in [torch.sin, torch.cos]:
            if stop_gradient:
                rets.append(fn(factor**i * x * mult_factor).detach())
            else:
                rets.append(fn(factor**i * x * mult_factor))
    # return torch.cat(rets, 1)
    # To make sure the dimensions of the same meaning are together
    return torch.flatten(torch.stack(rets, -1), start_dim=-2, end_dim=-1)


class PoseEnc(nn.Module):
    def __init__(self, factor=2.0, mult_factor=1.0):
        super(PoseEnc, self).__init__()
        self.factor = factor
        self.mult_factor = mult_factor

    def forward(self, x, L_embed, without_self=False, stop_gradient=False):
        return posenc(x, L_embed, self.factor, without_self, self.mult_factor, stop_gradient)


def normalize_vector(x, eps=1e-6):
    # assert(x.shape[-1] == 3)
    return x / (torch.norm(x, dim=-1, keepdim=True) + eps)


def create_learning_rate_fn(optimizer, max_steps, args, use_warmup=True, debug=False):
    """Create learning rate schedule."""
    if args.type == "none":
        return None

    if use_warmup and int(args.warmup) > 0:
        warmup_start_factor = 1e-8
        warmup_total_iters = int(args.warmup)
    else:
        warmup_start_factor = 1.0
        warmup_total_iters = 0

    warmup_fn = lr_scheduler.LinearLR(optimizer,
                                      start_factor=warmup_start_factor,
                                      end_factor=1.0,
                                      total_iters=warmup_total_iters)

    if args.type == "linear":
        decay_fn = lr_scheduler.LinearLR(optimizer,
                                         start_factor=1.0,
                                         end_factor=0.,
                                         total_iters=max_steps - warmup_total_iters)
        schedulers = [warmup_fn, decay_fn]
        milestones = [warmup_total_iters]

    elif args.type == "cosine":
        cosine_steps = max(max_steps - warmup_total_iters, 1)
        decay_fn = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_steps)
        schedulers = [warmup_fn, decay_fn]
        milestones = [warmup_total_iters]

    elif args.type == "cosine-hlfperiod":
        cosine_steps = max(max_steps - warmup_total_iters, 1) * 2
        decay_fn = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_steps)
        schedulers = [warmup_fn, decay_fn]
        milestones = [warmup_total_iters]

    elif args.type == "exp":
        decay_fn = lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
        schedulers = [warmup_fn, decay_fn]
        milestones = [warmup_total_iters]

    elif args.type == "stop":
        decay_fn = lr_scheduler.StepLR(optimizer, step_size=1)
        schedulers = [warmup_fn, decay_fn]
        milestones = [warmup_total_iters]

    else:
        raise NotImplementedError

    schedule_fn = lr_scheduler.SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)

    return schedule_fn


class Sine(nn.Module):
    def __init__(self, factor=30):
        super().__init__()
        self.factor = factor

    def forward(self, x):
        return torch.sin(x * self.factor)


class Clamp(nn.Module):
    def __init__(self, min_val, max_val):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        return torch.clamp(x, self.min_val, self.max_val)


class SHPlusHalfMin(nn.Module):
    def __init__(self, min_val=0):
        super().__init__()
        self.min_val = min_val

    def forward(self, x):
        return torch.clamp_min(x + 0.5, self.min_val)


class ShiftedTanh(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (torch.tanh(x) + 1) / 2


class SoftplusActivation(nn.Module):
    def __init__(self, c1=1, c2=1, c3=0):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3

    def forward(self, x):
        return self.c1 * nn.functional.softplus(self.c2 * x + self.c3)


class GaussianActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))

    def forward(self, x):
        return torch.exp(-x**2/(2*self.a**2))


class QuadraticActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))

    def forward(self, x):
        return 1/(1+(self.a*x)**2)


class MultiQuadraticActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))

    def forward(self, x):
        return 1/(1+(self.a*x)**2)**0.5


class LaplacianActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))

    def forward(self, x):
        return torch.exp(-torch.abs(x)/self.a)


class SuperGaussianActivation(nn.Module):
    def __init__(self, a=1., b=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))
        self.register_parameter('b', nn.Parameter(b*torch.ones(1), trainable))

    def forward(self, x):
        return torch.exp(-x**2/(2*self.a**2))**self.b


class ExpSinActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))

    def forward(self, x):
        return torch.exp(-torch.sin(self.a*x))


class PlusOneActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x + 1


class FlippedELU(nn.Module):
    def __init__(self, alpha=0.2):
        super(FlippedELU, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return -self.alpha * torch.nn.functional.elu(-x)
    

def sphere_pc(center, num_pts, scale):
    xs, ys, zs = [], [], []
    phi = math.pi * (3. - math.sqrt(5.))
    for i in range(num_pts):
        y = 1 - (i / float(num_pts - 1)) * 2
        radius = math.sqrt(1 - y * y)
        theta = phi * i
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        xs.append(x * scale[0] + center[0])
        ys.append(y * scale[1] + center[1])
        zs.append(z * scale[2] + center[2])
    points = np.stack([np.array(xs), np.array(ys), np.array(zs)], axis=-1)
    return torch.from_numpy(points).float()


def semi_sphere_pc(center, num_pts, scale, flatten="-z", flatten_coord=0.0):
    xs, ys, zs = [], [], []
    phi = math.pi * (3. - math.sqrt(5.))
    for i in range(num_pts):
        y = 1 - (i / float(num_pts - 1)) * 2
        radius = math.sqrt(1 - y * y)
        theta = phi * i
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        xs.append(x * scale[0] + center[0])
        ys.append(y * scale[1] + center[1])
        zs.append(z * scale[2] + center[2])
    points = np.stack([np.array(xs), np.array(ys), np.array(zs)], axis=-1)
    points = torch.from_numpy(points).float()
    if flatten == "-z":
        points[:, 2] = torch.clamp(points[:, 2], min=flatten_coord)
    elif flatten == "+z":
        points[:, 2] = torch.clamp(points[:, 2], max=flatten_coord)
    elif flatten == "-y":
        points[:, 1] = torch.clamp(points[:, 1], min=flatten_coord)
    elif flatten == "+y":
        points[:, 1] = torch.clamp(points[:, 1], max=flatten_coord)
    elif flatten == "-x":
        points[:, 0] = torch.clamp(points[:, 0], min=flatten_coord)
    elif flatten == "+x":
        points[:, 0] = torch.clamp(points[:, 0], max=flatten_coord)
    else:
        raise ValueError("Invalid flatten type")
    return points


def cube_pc(center, num_pts, scale):
    xs = np.random.uniform(-scale[0], scale[0], num_pts) + center[0]
    ys = np.random.uniform(-scale[1], scale[1], num_pts) + center[1]
    zs = np.random.uniform(-scale[2], scale[2], num_pts) + center[2]
    points = np.stack([np.array(xs), np.array(ys), np.array(zs)], axis=-1)
    return torch.from_numpy(points).float()


def cube_normal_pc(center, num_pts, scale):
    axis_num_pts = int(num_pts ** (1.0 / 3.0))
    xs = np.linspace(-scale[0], scale[0], axis_num_pts) + center[0]
    ys = np.linspace(-scale[1], scale[1], axis_num_pts) + center[1]
    zs = np.linspace(-scale[2], scale[2], axis_num_pts) + center[2]
    points = np.array([[i, j, k] for i in xs for j in ys for k in zs])
    rest_num_pts = num_pts - points.shape[0]
    if rest_num_pts > 0:
        rest_points = cube_pc(center, rest_num_pts, scale)
        points = np.concatenate([points, rest_points], axis=0)
    return torch.from_numpy(points).float()


def construct_coord_frame(
        z: T.Union[np.ndarray, torch.Tensor] = (0, 0, -1.),
        y: T.Union[np.ndarray, torch.Tensor] = (0, -1., 0.),
) -> T.Union[np.ndarray, torch.Tensor]:
    """
    Get a coordinate frame from z and y vector.
    z will be used directly as the z axis.
    y will be made orthogonal to y and used as the y axis.
    x axis will be the the cross-product of y axis and z axis.
    All axes are normalised to have unit norm.

    Args:
        z: (*, 3)
        y: (*, 3)

    Returns:
        (*, 3, 3):
        For the last 2 dimension, the first column is the x axis, second y, last z.
        It can be used as the rotation matrix that transform
        a vector in camera coord to world coord.
    """

    if isinstance(z, (tuple, list)):
        z = torch.tensor(z)
    if isinstance(y, (tuple, list)):
        y = torch.tensor(y)

    is_numpy = False
    if isinstance(z, np.ndarray):
        z = torch.from_numpy(z)
        is_numpy = True
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y)
        is_numpy = True

    z_norm = torch.linalg.norm(z, ord=2, dim=-1, keepdim=True)  # (*, 1)
    assert torch.all(z_norm > 0)
    assert torch.all(torch.linalg.norm(y, ord=2, dim=-1) > 0)
    x = torch.cross(y, z, dim=-1)  # (*, 3)
    if torch.any(torch.linalg.norm(x, ord=2, dim=-1) == 0):
        raise ValueError("y and z cannot be parallel.")

    # make sure y-axis is perpendicular to z-axis
    z = z / z_norm  # (*, 3)
    y_on_z = torch.sum(y * z, dim=-1, keepdim=True) * z  # (*, 3)
    y = y - y_on_z

    # normalize
    y = y / torch.linalg.norm(y, ord=2, dim=-1, keepdim=True)  # (*, 3)
    x = x / torch.linalg.norm(x, ord=2, dim=-1, keepdim=True)  # (*, 3)

    Rs = torch.stack((x, y, z), dim=-1)  # (*, 3, 3)

    if is_numpy:
        Rs = Rs.detach().cpu().numpy()

    return Rs


def rectify_points(
        points: torch.Tensor,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        translate: bool = False,
        randomize_translate: bool = False,
        ts: torch.Tensor = None,
        t_min: float = 0.,
        t_max: float = 1e6,
):
    """
    Given n points associated with each of the m rays (each row in points),
    rotate and translate the coordinate so that
    - ray direction becomes (0,0,1)
    - ray origin becomes (0,0,0)
    - if translate is True, the coordinate origin is chosen so that the t to the closest projection is 1

    Args:
        points:
            (*, m, n, 3)  xyz
        ray_origins:
            (*, m, 3)
        ray_directions:
            (*, m, 3)
        translate:
            whether to translate the coord so that the closest point's projection on the
            ray has t = 0 for all t > 0.
        randomize_translate:
            only used when translate is true.
            If randomize_translate = true, tt will be one random distance between 0 and closest point
        ts:
            (*, m, n)  the projection length each points on the ray
            should be given only if translate is True.

    Returns:
        points_n:
            (*, m, n, 3)  the transformed points
        Rs_w2n:
            (*, m, 3, 3) the rotation matrix that transform the world coord to the rectified coord
        translation_w2n:
            (*, m, 3, 1) the translation vector that transform the world coord to the rectified coord
        tt:
            (*, m) the t that we subtract from the input ts
    """
    y = torch.zeros_like(ray_directions)  # (*, m, 3)
    y[..., 1] = 1.  # try to use the current y-axis as the y-axis
    Rs_n2w = construct_coord_frame(
        z=ray_directions,
        y=y,
    )  # (*, m, 3, 3)  the column in the 3*3 matrix is the axis coord with unit norm
    # Rs_n2w = torch.randn(*(ray_directions.shape[:-1]), 3, 3, device=ray_directions.device)
    # it can be thought of as the transform matrix from the new coord to the world coord

    if translate:
        assert ts is not None
        ts = ts.clone()
        ts[torch.logical_or(
            ts < t_min,
            ts > t_max)] = torch.inf  # we do not care about negative ts (potential problem: all ts could be neg)
        tt, _ = ts.min(dim=-1, keepdim=True)  # (*, m, 1)
        tt[~torch.isfinite(tt)] = 0  # if all ts are neg, do not move the origin
        # in pr, invalid point is represented by background far-away point
        # do not shift if all points are background point as well
        # note that training process could occassionaly have a large hit error (should be OK since gradient is clipped)
        # maybe it is because far_thres 1e6 too large to thres out some points...?

        tt[tt > t_max] = 0
        if randomize_translate:
            # tt = tt * torch.rand(tt.shape, device=tt.device)
            tt = tt - torch.rand(tt.shape, device=tt.device) * 0.1
        else:
            pass
            # tt = tt - 0.05

        origins_w = ray_origins + tt * ray_directions  # (*, m, 3)
    else:
        tt_shape = list(ray_origins.shape)
        tt_shape[-1] = 1
        tt = torch.zeros(*tt_shape, device=ray_origins.device)  # (*, m, 1)
        origins_w = ray_origins  # (*, m, 3)

    # create H_w2n (note the inversion)
    Rs_w2n = Rs_n2w.transpose(-1, -2)  # (*, m, 3, 3)
    translation_w2n = -1.0 * (Rs_w2n @ origins_w.unsqueeze(-1))  # (*, m, 3, 1)

    # transform the points (m, n, 3):  (*, m, 1, 3, 3) @ (*, m, n, 3, 1) + (*, m, 1, 3, 1)
    points_n = Rs_w2n.unsqueeze(-3) @ points.unsqueeze(-1) + translation_w2n.unsqueeze(-3)  # (*, m, n, 3, 1)

    return dict(
        points_n=points_n.squeeze(-1),  # (*, m, n, 3)
        Rs_w2n=Rs_w2n,  # (*, m, 3, 3)
        translation_w2n=translation_w2n,  # (*, m, 3, 1)
        tt=tt.squeeze(-1),  # (*, m) the t that we subtract from the input ts
    )
    
    
def scale_scene_to_unit_box(points, rays_o, rays_d):
    """
    Scales a zero-centered point cloud and associated rays to fit within [-1, 1].
    
    Args:
        points: (N, 3) Point cloud (assumed zero-centered)
        rays_o: (M, 3) Ray Origins
        rays_d: (M, 3) Ray Directions
        
    Returns:
        points_scaled: (N, 3) Points in [-1, 1]
        rays_o_scaled: (M, 3) Ray origins scaled to new space
        rays_d_norm:   (M, 3) Normalized ray directions (Unit vectors)
        scale_factor:  (float) The value used to scale (1.0 / max_extent)
    """
    # 1. Calculate Scale Factor
    # We find the furthest point from the origin in any dimension.
    # We add a tiny epsilon (1e-8) to avoid division by zero if points are all 0.
    max_abs_coord = torch.abs(points).max()
    scale_factor = 1.0 / (max_abs_coord + 1e-8)
    
    # Optional: Slightly shrink scale (e.g. 0.99) to ensure points 
    # don't hit the exact boundary of the voxel grid, which can cause edge cases.
    scale_factor = scale_factor * 0.999
    
    # 2. Scale Geometry (Positions)
    # Positions (Points and Origins) scale linearly.
    points_scaled = points * scale_factor
    rays_o_scaled = rays_o * scale_factor
    
    # 3. Handle Directions
    # Directions represent orientation. If we scale the magnitude of rays_d,
    # it stops being a unit vector, which often breaks ray tracers (like Kaolin).
    # We re-normalize them just to be safe.
    rays_d_norm = rays_d / (torch.norm(rays_d, dim=-1, keepdim=True) + 1e-8)
    
    return points_scaled, rays_o_scaled, rays_d_norm, scale_factor


def create_boundary_mask(foreground_mask, boundary_width=1):
    """
    Create a mask where boundary pixels are zeros and non-boundary pixels are ones.
    
    The boundary is defined as the region where foreground meets background.
    This is computed by eroding the foreground mask and finding the difference.
    
    Args:
        foreground_mask: (H, W) or (B, H, W) or (B, 1, H, W) tensor/array
                        where foreground pixels are 1 and background pixels are 0
        boundary_width: int, the width of the boundary in pixels (default: 1)
    
    Returns:
        boundary_mask: Same shape as input, where boundary pixels are 0 and 
                      non-boundary pixels are 1
    """
    is_numpy = isinstance(foreground_mask, np.ndarray)
    if is_numpy:
        foreground_mask = torch.from_numpy(foreground_mask)
    
    original_shape = foreground_mask.shape
    original_dtype = foreground_mask.dtype

    # Ensure the mask is float for operations
    mask = foreground_mask.float()
    
    # Handle different input dimensions
    # We need (B, C, H, W) for conv2d
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)  # (H, W) -> (1, 1, H, W)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
    # else: assume (B, C, H, W)
    
    # Erosion and dilation are done with min/max pooling below, so only the
    # window size is needed.
    kernel_size = 2 * boundary_width + 1

    # Erode the foreground mask using min pooling (via negative max pooling)
    # Erosion: pixel is 1 only if all pixels in the neighborhood are 1
    eroded = -torch.nn.functional.max_pool2d(
        -mask, 
        kernel_size=kernel_size, 
        stride=1, 
        padding=boundary_width
    )
    
    # Also erode the background (i.e., dilate the foreground)
    # to get the outer boundary
    dilated = torch.nn.functional.max_pool2d(
        mask,
        kernel_size=kernel_size,
        stride=1,
        padding=boundary_width
    )
    
    # Boundary is where (dilated - eroded) > 0
    # i.e., pixels that are in the dilated but not in the eroded version
    boundary = (dilated - eroded) > 0
    
    # Create the output mask: 1 for non-boundary, 0 for boundary
    boundary_mask = (~boundary).float()
    
    # Reshape back to original shape
    if len(original_shape) == 2:
        boundary_mask = boundary_mask.squeeze(0).squeeze(0)
    elif len(original_shape) == 3:
        boundary_mask = boundary_mask.squeeze(1)
    
    # Convert back to original dtype
    boundary_mask = boundary_mask.to(original_dtype)
    
    if is_numpy:
        boundary_mask = boundary_mask.numpy()
    
    return boundary_mask


def dilate_mask(mask, kernel_size=5, iterations=1):
    """
    PyTorch implementation of morphological dilation for batched binary masks.
    Equivalent to cv2.dilate but works on GPU and supports batch operations.
    
    Args:
        mask: Boolean tensor of shape (N, H, W) or (H, W)
        kernel_size: Size of the dilation kernel (default: 5)
        iterations: Number of dilation iterations (default: 1)
    
    Returns:
        Dilated boolean mask with same shape as input
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
    
    # Apply dilation iterations
    # Morphological dilation: output = 1 if ANY pixel in kernel neighborhood is 1, else 0
    # This is equivalent to max pooling
    result = mask_float
    for _ in range(iterations):
        result = result.unsqueeze(1)  # (N, 1, H, W) for conv operations
        
        # Dilation: max pooling
        result = F.max_pool2d(result, kernel_size=kernel_size, stride=1, padding=padding)
        
        result = result.squeeze(1)  # (N, H, W)
    
    # Convert back to boolean
    result = result > 0.5
    
    if squeeze_output:
        result = result.squeeze(0)
    
    return result


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


def erode_mask_by_pixel_count(mask, pixels_to_erode, kernel_size=3, max_iterations=None):
    """
    Erode a binary mask until at least `pixels_to_erode` foreground pixels are removed.

    This gives direct control over erosion strength in pixel count space instead of
    kernel-size/iteration space.

    Args:
        mask: Boolean tensor of shape (N, H, W) or (H, W)
        pixels_to_erode: Target number of foreground pixels to remove per mask
        kernel_size: Kernel size used per erosion step (default: 3)
        max_iterations: Safety cap for erosion loop (default: max(H, W))

    Returns:
        Eroded boolean mask with same shape as input
    """
    if pixels_to_erode <= 0:
        return mask.clone()

    # Handle single image case
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    padding = kernel_size // 2
    result = mask.bool().clone()
    start_count = result.sum(dim=(-2, -1))
    target_count = torch.clamp(start_count - int(pixels_to_erode), min=0)

    if max_iterations is None:
        max_iterations = max(result.shape[-2], result.shape[-1])

    for _ in range(max_iterations):
        current_count = result.sum(dim=(-2, -1))
        active = current_count > target_count
        if not torch.any(active):
            break

        # One-step erosion inline (min-pooling via max-pooling on inverted mask)
        result_float = result.float().unsqueeze(1)  # (N, 1, H, W)
        next_result_inv = F.max_pool2d(1 - result_float, kernel_size=kernel_size, stride=1, padding=padding)
        next_result = (1 - next_result_inv).squeeze(1) > 0.5
        next_count = next_result.sum(dim=(-2, -1))

        # Only update masks that still need erosion and actually changed.
        updatable = active & (next_count < current_count)
        if not torch.any(updatable):
            break
        result[updatable] = next_result[updatable]

    if squeeze_output:
        result = result.squeeze(0)

    return result


def get_boundary_exclusion_mask(fg_mask, erode_kernel_size=5, dilate_kernel_size=5, erode_iterations=1, dilate_iterations=1):
    """
    Create a mask that excludes boundary regions.
    The mask is True on both foreground and background, but False near the boundary.
    
    This is computed as: eroded_fg_mask OR (NOT dilated_fg_mask)
    - eroded_fg_mask: foreground shrunk inward (removes outer boundary of foreground)
    - NOT dilated_fg_mask: background shrunk inward (removes outer boundary of background, i.e., inner boundary near foreground)
    
    The result is True for interior foreground AND interior background, False near boundaries.
    
    Args:
        fg_mask: Boolean tensor of shape (N, H, W) or (H, W)
        erode_kernel_size: Size of the erosion kernel (default: 5)
        dilate_kernel_size: Size of the dilation kernel (default: 5)
        erode_iterations: Number of erosion iterations (default: 1)
        dilate_iterations: Number of dilation iterations (default: 1)
    
    Returns:
        Boolean mask with same shape as input, True for non-boundary regions
    """
    eroded_fg = erode_mask(fg_mask, kernel_size=erode_kernel_size, iterations=erode_iterations)
    dilated_fg = dilate_mask(fg_mask, kernel_size=dilate_kernel_size, iterations=dilate_iterations)
    
    # Boundary exclusion mask: interior foreground OR interior background
    # Interior foreground = eroded_fg_mask
    # Interior background = NOT dilated_fg_mask
    boundary_exclusion_mask = eroded_fg | (~dilated_fg)
    
    return boundary_exclusion_mask
