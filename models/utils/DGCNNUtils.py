import torch

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