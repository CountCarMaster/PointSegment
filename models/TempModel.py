from models.utils.TempModelUtils import *
import torch.nn as nn
import torch.nn.functional as F

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
        mask = centroids[:, :, :, 2] > 0
        mask = mask * 2 - 1
        normals *= mask.unsqueeze(-1)
        feature = torch.cat((centroids, normals), dim=-1)
        feature = torch.mean(feature, dim=-2)
        return feature

class TempModel(nn.Module):
    def __init__(self, num_classes):
        super(TempModel, self).__init__()
        self.sa1 = PointNetSetAbstraction(32, 32, 1024, 3 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(32, 32, 256, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(32, 32, 64, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(32, 32, 16, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)\

        self.geometric_extractor = GeometricExtractor()

    def forward(self, data):
        l0_points = data
        l0_xyz = data[:,:3,:]

        l0_xyz = l0_xyz.transpose(1, 2)
        l0_points = l0_points.transpose(1, 2)

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points.transpose(1, 2))
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points.transpose(1, 2))
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points.transpose(1, 2))

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        # x = x.permute(0, 2, 1)
        return x