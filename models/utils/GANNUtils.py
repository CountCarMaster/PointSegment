import torch
import numpy as np
import torch.nn as nn

def knn(x, k):
    """
    get k nearest neighbors.
    :param x: input torch tensors, size (B, N, C)
    :param k: neighbor size.
    :return:
    """
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

def keep_feature_local(x, k, normalize='center'):
    B, channel, N = x.shape
    idx = knn(x, k).transpose(1, 2)
    x_expand = x.unsqueeze(2).expand(B, channel, k, N)
    idx_expand = idx.unsqueeze(1).expand(B, channel, k, N)
    neighbors = torch.gather(x_expand, 3, idx_expand)
    neighbors = neighbors.transpose(1, 3)
    if normalize is not None:
        affine_alpha = nn.Parameter(torch.ones([1, 1, 1, channel])).to(x.device)
        affine_beta = nn.Parameter(torch.zeros([1, 1, 1, channel])).to(x.device)
    mean = None
    if normalize == "center":
        mean = torch.mean(neighbors, dim=2, keepdim=True)
    if normalize == "anchor":
        mean = neighbors
        mean = mean.unsqueeze(dim=-2)  # [B, npoint, 1, d+3]
    std = torch.std((neighbors - mean).reshape(B, -1), dim=-1, keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
    neighbors = (neighbors - mean) / (std + 1e-5)
    neighbors = affine_alpha * neighbors + affine_beta
    return neighbors.transpose(1, 3)

def get_graph_feature(x, k):
    """
    aggressive the neighbor feature
    :param x: input torch tensors, size (B, N, C)
    :param k: k nearest neighbors.
    :return:
    """
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

class AddNorm(nn.Module):
    def __init__(self, dim):
        super(AddNorm, self).__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(self.dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input, x):
        input = self.dropout(input)
        return self.norm(input + x)

class AttnKnn(nn.Module):
    def __init__(self, num_points, in_channel, qkv_channel, k_tmp=50, k=20):
        super(AttnKnn, self).__init__()
        self.k_tmp = k_tmp
        self.k = k
        self.N = num_points

        # Embedding
        self.q_linear = nn.Linear(in_channel, qkv_channel)
        self.k_linear = nn.Linear(in_channel, qkv_channel)
        self.v_linear = nn.Linear(in_channel, in_channel)
        self.fc = nn.Linear(in_channel, in_channel)
        self.dense = nn.Linear(self.N, self.N)
        self.in_channel = in_channel

        # add norm
        self.add_norm1 = AddNorm(self.N)
        self.add_norm2 = AddNorm(self.N)

    def forward(self, x):
        B, C, N = x.shape  # [B, C, N]

        x_transposed = x.transpose(1, 2)  # [B, N, C]

        # Aggressive neighbor feature
        x_with_neighbour = get_graph_feature(x, self.k_tmp)  # [B, N, k, C]
        q = self.q_linear(x_transposed)  # [B, N, C']
        k = self.k_linear(x_with_neighbour)  # [B, N, k, C']
        v = self.v_linear(x_with_neighbour)
        scaled_attention_logits = torch.matmul(q.unsqueeze(2), k.transpose(-2, -1).contiguous())
        scaled_attention_logits = scaled_attention_logits.view(B, N, -1)
        scaled_attention_logits /= torch.sqrt(torch.tensor(self.N, dtype=torch.float32))
        attention_weights = torch.nn.functional.softmax(scaled_attention_logits, dim=-1)
        _, indices = torch.sort(attention_weights, dim=-1)
        indices = indices[:, :, :self.k]  # [B, N, k]
        output = attention_weights.unsqueeze(-1) * v
        output = self.add_norm1(output.view(B, -1, N), x_with_neighbour.view(B, -1, N))
        x = self.dense(output)
        x = self.add_norm2(x, output)
        x = x.view(B, -1, C, self.N).contiguous()
        x += x_with_neighbour.permute(0, 2, 3, 1)
        x = x.permute(0, 3, 1, 2)
        x = self.fc(x)
        ans = torch.gather(x, dim=-2, index=indices.unsqueeze(-1).expand(-1, -1, -1, C))
        # print(ans.permute(0, 3, 2, 1).shape)
        return ans.permute(0, 3, 2, 1)  # [B, C, k, N]

class AttnKnnWeightOnly(nn.Module):
    def __init__(self, num_points, in_channel, qkv_channel, k=20):
        super(AttnKnnWeightOnly, self).__init__()
        self.k = k
        self.N = num_points

        # Embedding
        self.q_linear = nn.Linear(in_channel, qkv_channel)
        self.k_linear = nn.Linear(in_channel, qkv_channel)
        self.v_linear = nn.Linear(in_channel, in_channel)
        self.fc = nn.Linear(in_channel, in_channel)
        self.dense = nn.Linear(self.N, self.N)
        self.in_channel = in_channel

        # add norm
        self.add_norm1 = AddNorm(self.N)
        self.add_norm2 = AddNorm(self.N)

    def forward(self, x):
        B, C, N = x.shape  # [B, C, N]

        x_transposed = x.transpose(1, 2)  # [B, N, C]

        # Aggressive neighbor feature
        x_with_neighbour = get_graph_feature(x, self.k)  # [B, N, k, C]
        q = self.q_linear(x_transposed)  # [B, N, C']
        k = self.k_linear(x_with_neighbour)  # [B, N, k, C']
        v = self.v_linear(x_with_neighbour)
        scaled_attention_logits = torch.matmul(q.unsqueeze(2), k.transpose(-2, -1).contiguous())
        scaled_attention_logits = scaled_attention_logits.view(B, N, -1)
        scaled_attention_logits /= torch.sqrt(torch.tensor(self.N, dtype=torch.float32))
        attention_weights = torch.nn.functional.softmax(scaled_attention_logits, dim=-1)
        output = attention_weights.unsqueeze(-1) * v
        output = self.add_norm1(output.view(B, -1, N), x_with_neighbour.view(B, -1, N))
        x = self.dense(output)
        x = self.add_norm2(x, output)
        x = x.view(B, -1, C, self.N).contiguous()
        x += x_with_neighbour.permute(0, 2, 3, 1)
        x = x.permute(0, 3, 1, 2)
        x = self.fc(x)
        # ans = torch.gather(x, dim=-2, index=indices.unsqueeze(-1).expand(-1, -1, -1, C))
        # print(x.permute(0, 3, 2, 1).shape)
        return x.permute(0, 3, 2, 1)