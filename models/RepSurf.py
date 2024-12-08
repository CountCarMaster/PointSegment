import torch
import torch.nn as nn
from models.utils.RepSurfUtils import keep_feature, get_graph_feature, resort_points, xyz2sphere

class GeometricExtractor(nn.Module):
    def __init__(self):
        super(GeometricExtractor, self).__init__()
        self.bn1 = nn.BatchNorm1d(10)
        self.bn2 = nn.BatchNorm1d(10)
        self.mlp1 = nn.Sequential(
            nn.Linear(7, 10),
            self.bn1,
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(10, 10),
            self.bn2,
            nn.ReLU()
        )

    def forward(self, x, k=9):
        B, N, C = x.shape
        device = x.device
        x_expand = torch.unsqueeze(x, -2).to(device)
        x_neighbour = get_graph_feature(x.transpose(1, 2), k + 1) - x_expand
        x_neighbour = x_neighbour[:, :, 1:, :]
        x_sphere = xyz2sphere(x_neighbour)
        phi = x_sphere[..., 2]
        idx = phi.argsort(dim=-1)
        pairs = resort_points(x_neighbour, idx).unsqueeze(-2)
        pairs = torch.cat((pairs, torch.roll(pairs, -1, dims=-3)), dim=-2)
        centroids = torch.mean(pairs, dim=-2)
        vector_1 = pairs[..., 0, :].view(B, N, k, -1)
        vector_2 = pairs[..., 1, :].view(B, N, k, -1)
        normals = torch.cross(vector_1, vector_2, dim=-1)  # [B, N, k, 3]
        normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-6)
        pos_mask = (normals[..., 0:1, 0] > 0).float() * 2. - 1.
        normals *= torch.unsqueeze(pos_mask, -1)
        position = torch.sum(normals * centroids, dim=-1) / torch.sqrt(torch.tensor(3))
        position = position.unsqueeze(-1)
        feature = torch.cat((centroids, normals, position), dim=-1)
        feature = feature.view(B * N * k, -1)

        feature = self.mlp1(feature)
        feature = self.mlp2(feature)
        feature = feature.view(B, N, k, -1)
        feature = torch.max(feature, dim=-2)[0]
        return feature

class RepSurf(nn.Module):
    def __init__(self, input_channel, output_channel, k=5):
        super(RepSurf, self).__init__()
        self.k = k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm1d(1024)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(128)

        self.geometric_extractor = GeometricExtractor()

        self.edge_cov1 = nn.Sequential(
            nn.Conv2d(10, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.ReLU()
        )
        self.edge_cov2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.ReLU()
        )
        self.edge_cov3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            self.bn3,
            nn.ReLU()
        )
        self.edge_cov4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            self.bn4,
            nn.ReLU()
        )
        self.edge_cov5 = nn.Sequential(
            nn.Conv1d(64, 1024, kernel_size=1, bias=False),
            self.bn5,
            nn.ReLU()
        )
        self.edge_cov6 = nn.Sequential(
            nn.Conv1d(1088, 512, kernel_size=1, bias=False),
            self.bn6,
            nn.ReLU()
        )
        self.edge_cov7 = nn.Sequential(
            nn.Conv1d(512, 128, kernel_size=1, bias=False),
            self.bn7,
            nn.ReLU()
        )
        self.edge_cov8 = nn.Sequential(
            nn.Conv1d(128, output_channel, kernel_size=1, bias=False),
            nn.ReLU()
        )

    def forward(self, x): #[B, C, N]
        B, C, N = x.shape

        x = self.geometric_extractor(x.transpose(1, 2)).transpose(1, 2)

        x = keep_feature(x, self.k) # [B, C, k, N]
        x = self.edge_cov1(x)
        x = torch.max(x, dim=2)[0]

        x = keep_feature(x, self.k)
        x = self.edge_cov2(x)
        x1 = torch.max(x, dim=2)[0]

        x = keep_feature(x1, self.k)
        x = self.edge_cov3(x)
        x = torch.max(x, dim=2)[0]

        x = keep_feature(x, self.k)
        x = self.edge_cov4(x)
        x = torch.max(x, dim=2)[0]

        x = self.edge_cov5(x)
        x = torch.max(x, dim=2)[0]

        x = torch.cat([x1, x.unsqueeze(-1).repeat(1, 1, N)], dim=1)

        x = self.edge_cov6(x)
        x = self.edge_cov7(x)
        x = self.edge_cov8(x)
        return x