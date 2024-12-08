import torch
import numpy as np

def knn(x, k):
    x_square = torch.sum(x ** 2, dim=1, keepdim=True)
    xx = torch.matmul(x.transpose(2, 1), x)
    distance = -x_square - x_square.transpose(2, 1) + 2 * xx
    idx = distance.topk(k=k, dim=-1)[1]
    return idx

def keep_feature(x, k):
    B, C, N = x.shape
    idx = knn(x, k).transpose(1, 2)
    x_expand = x.unsqueeze(2).expand(B, C, k, N)
    idx_expand = idx.unsqueeze(1).expand(B, C, k, N)
    neighbors = torch.gather(x_expand, 3, idx_expand)
    neighbors += x_expand
    return neighbors

def get_graph_feature(x, k):
    device = x.device
    batch_size = x.size(0)
    num_dims = x.size(1)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    idx = knn(x, k).to(device)
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    x = x.transpose(2, 1).contiguous().to(device)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    return feature

def xyz2sphere(xyz, normalize=True):
    rho = torch.sqrt(torch.sum(torch.pow(xyz, 2), dim=-1, keepdim=True))
    rho = torch.clamp(rho, min=0)  # range: [0, inf]
    theta = torch.acos(xyz[..., 2, None] / rho)  # range: [0, pi]
    phi = torch.atan2(xyz[..., 1, None], xyz[..., 0, None])  # range: [-pi, pi]
    # check nan
    idx = rho == 0
    theta[idx] = 0

    if normalize:
        theta = theta / np.pi  # [0, 1]
        phi = phi / (2 * np.pi) + .5  # [0, 1]
    out = torch.cat([rho, theta, phi], dim=-1)
    return out


def resort_points(points, idx):
    device = points.device
    B, N, G, _ = points.shape

    view_shape = [B, 1, 1]
    repeat_shape = [1, N, G]
    b_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)

    view_shape = [1, N, 1]
    repeat_shape = [B, 1, G]
    n_indices = torch.arange(N, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)

    new_points = points[b_indices, n_indices, idx, :]

    return new_points