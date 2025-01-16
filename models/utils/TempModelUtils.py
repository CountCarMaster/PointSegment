import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def knn(x, k):
    x_square = torch.sum(x ** 2, dim=1, keepdim=True)
    xx = torch.matmul(x.transpose(2, 1), x)
    distance = -x_square - x_square.transpose(2, 1) + 2 * xx
    idx = distance.topk(k=k, dim=-1)[1]
    return idx

def square_dist(a, b) :
    """
    Input:
        a: source, matrix sized [B, N, C]
        b: target, matrix sized [B, M, C]
    Output:
        ans: distance matrix, sized [B, N, M]
    """
    B, N, _ = a.shape
    _, M, _ = b.shape
    ans = -2 * torch.matmul(a, b.permute(0, 2, 1))
    # ans = -2 * torch.matmul(a, b.transpose(2, 1))
    ans += torch.sum(a ** 2, -1).view(B, N, 1)
    ans += torch.sum(b ** 2, -1).view(B, 1, M)
    return ans

def furthest_point_sampling(data, point_num) :
    """
    Input:
        data: data point sized [B, N, C]
        point_num: number of clusters
    Output:
        centroids: cluster centroids, sized [B, point_num]
    """
    device = data.device
    batch_size, num_points, num_channels = data.shape
    centroids = torch.zeros(batch_size, point_num, dtype=torch.long).to(device)
    dist = torch.ones(batch_size, num_points).to(device) * 1e10
    farthest = torch.randint(0, num_points, (batch_size,), dtype=torch.long).to(device)
    batch_indices = torch.arange(batch_size, dtype=torch.long).to(device)

    for i in range(point_num) :
        centroids[:, i] = farthest
        centroid = data[batch_indices, farthest, :].view(batch_size, 1, num_channels)
        dist_tmp = torch.sum((data - centroid) ** 2, -1)
        mask = dist_tmp < dist
        dist[mask] = dist_tmp[mask]
        farthest = torch.max(dist, -1)[1]
    return centroids

def index(data, idx) :
    """
    Input:
        data: data point sized [B, N, C]
        idx: wanted index of points, sized [B, S]
    Output:
        ans: indexed data points, sized [B, S, C]
    """
    device = data.device
    batch_size = data.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(batch_size, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    ans = data[batch_indices, idx, :]  # [B, N, k, c]
    return ans

def knn_query(data, query_data, max_neighbors, k) :
    """
    Input:
        data: data point sized [B, N, C]
        qdata: query point sized [B, S, C]
        max_neighbors: number of max neighbors
        r: radius of ball
    Output:
        ans: index of neighbors, sized [B, S, max_neighbors]
    """
    device = data.device
    batch_size, num_points, num_channels = data.shape
    _, query_points, _ = query_data.shape
    ans = torch.arange(num_points, dtype=torch.long).to(device).view(1, 1, num_points).repeat(batch_size, query_points, 1)
    dist = square_dist(query_data, data)
    idx = dist.topk(k=k*2, dim=-1)[1]
    return idx

def knn_query_2k(data, query_data, max_neighbors, k) :
    """
    Input:
        data: data point sized [B, N, C]
        qdata: query point sized [B, S, C]
        max_neighbors: number of max neighbors
        r: radius of ball
    Output:
        ans: index of neighbors, sized [B, S, max_neighbors]
    """
    device = data.device
    batch_size, num_points, num_channels = data.shape
    _, query_points, _ = query_data.shape
    ans = torch.arange(num_points, dtype=torch.long).to(device).view(1, 1, num_points).repeat(batch_size, query_points, 1)
    dist = square_dist(query_data, data)
    idx = dist.topk(k=k*2, dim=-1)[1]
    return idx

def sample_group(k, max_neighbors, data, cluster_num, data_feature, return_fps=False) :
    """
    Input:
        r: radius of ball
        max_neighbors: number of max neighbors
        data: data point sized [B, N, C]
        cluster_num: number of clusters
        data_feature: feature without position message sized [B, N, D]
    Output:
        centroids: cluster centroids, sized [B, cluster_num, C]
        ans: sampled points with position message, sized [B, cluster_num, max_neighbors, C + D]
        sample: sampled points without position message, sized [B, cluster_num, max_neighbors, C]
        fpsIdx: index of cluster centroid points, sized [B, cluster_num]

    """
    batch_size, num_points, num_channels = data.shape
    fps_idx = furthest_point_sampling(data, cluster_num)
    centroids = index(data, fps_idx) # [B, cluster_num, C]
    sample_idx = knn_query_2k(data, centroids, max_neighbors, k)
    sample = index(data, sample_idx) # [B, N, k, C]

    diff_data = torch.zeros_like(sample)
    diff_data[:, :, 1:, :] = sample[:, :, 1:, :] - sample[:, :, :-1, :]
    diff_data[:, :, 0, :] = sample[:, :, 0, :] - sample[:, :, -1, :]
    # score = torch.sum(torch.abs(diff_data), dim=-1)
    score = torch.abs(diff_data)[:, :, :, 2]
    top_k_values, top_k_indices = torch.topk(score, k, dim=-1, largest=True)
    sample_idx = top_k_indices
    sample = torch.gather(sample, dim=2,
                                 index=top_k_indices.unsqueeze(-1).expand(-1, -1, -1, sample.size(-1)))

    sample_norm = sample - centroids.view(batch_size, cluster_num, 1, num_channels)

    if data_feature is not None:
        tmp = index(data_feature, sample_idx)
        ans = torch.cat([sample_norm, tmp], dim=-1)
        z_cha = sample[:, :, :, -1] - centroids.unsqueeze(-2)[:, :, :, -1]
        z_cha = torch.abs(z_cha)
        z_cha = torch.softmax(z_cha, dim=-1)
        ans *= z_cha.unsqueeze(-1)
    else :
        ans = sample_norm

    if return_fps:
        return centroids, ans, sample, fps_idx
    else:
        return centroids, ans

def sample_group_all(data, data_feature):
    """
    Input:
        data: data point sized [B, N, C]
        pdata: feature without position message sized [B, N, D]
    Output:
        centroids: cluster centroids, sized [B, 1, C]
        ans: sampled points with position message, sized [B, 1, N, C + D]
    """
    device = data.device
    batch_size, num_points, num_channels = data.shape
    centroids = torch.zeros(batch_size, 1, num_channels).to(device)
    sample = data.view(batch_size, 1, num_points, num_channels)
    if data_feature is not None:
        new_points = torch.cat([sample, data_feature.view(batch_size, 1, num_points, -1)], dim=-1)
        return centroids, new_points
    else :
        return centroids, sample

class PointNetSetAbstraction(nn.Module):
    def __init__(self, k, max_neighbors, cluster_num, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.k = k
        self.cluster_num = cluster_num
        self.max_neighbors = max_neighbors
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, data, data_feature):
        """
        Input:
            data: data point sized [B, N, C]
            pdata: feature without position message sized [B, N, D]
        Output:
            centroids: cluster centroids, sized [B, npoints, C]
            sample: sampled points without position message, sized [B, npoints, C]
        """
        if self.group_all:
            centroids, sample = sample_group_all(data, data_feature)
        else :
            centroids, sample = sample_group(self.k, self.max_neighbors, data, self.cluster_num, data_feature)
        # [B, cluster_num, max_neighbors, C + D]
        sample = sample.permute(0, 3, 2, 1)
        # [B, C + D, max_neighbors, cluster_num]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            sample = F.relu(bn(conv(sample)))
        sample = torch.max(sample, 2)[0]
        # [B, C, cluster_num]
        return centroids, sample.permute(0, 2, 1)

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        # points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_dist(xyz1.transpose(1, 2), xyz2.transpose(1, 2))
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index(points2.transpose(1, 2), idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            # points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points

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