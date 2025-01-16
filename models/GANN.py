import torch
import torch.nn as nn
from models.utils.GANNUtils import keep_feature, keep_feature_local
from models.utils.GANNUtils import AttnKnnWeightOnly as AttnKnn

class GANN(nn.Module):
    def __init__(self, input_channel, output_channel, k=32):
        super(GANN, self).__init__()
        self.k = k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm1d(1024)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(128)

        self.knn1 = AttnKnn(in_channel=input_channel, qkv_channel=input_channel, num_points=2048, k=k)
        self.knn2 = AttnKnn(in_channel=64, qkv_channel=32, num_points=2048, k=k)
        self.knn3 = AttnKnn(in_channel=64, qkv_channel=32, num_points=2048, k=k)
        self.knn4 = AttnKnn(in_channel=64, qkv_channel=32, num_points=2048, k=k)


        self.edge_cov1 = nn.Sequential(
            nn.Conv2d(input_channel, 64, kernel_size=1, bias=False),
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

        x = keep_feature_local(x, self.k) # [B, C, k, N]
        x = self.edge_cov1(x)
        x = torch.max(x, dim=2)[0]

        # x0 = x.unsqueeze(2).repeat(1, 1, self.k, 1)
        x = keep_feature_local(x, self.k)
        # x = torch.cat((x, x0), dim=1)
        x = self.edge_cov2(x)
        x1 = torch.max(x, dim=2)[0]

        # x0 = x1.unsqueeze(2).repeat(1, 1, self.k, 1)
        x = keep_feature_local(x1, self.k)
        # x = torch.cat((x, x0), dim=1)
        x = self.edge_cov3(x)
        x = torch.max(x, dim=2)[0]

        # x0 = x.unsqueeze(2).repeat(1, 1, self.k, 1)
        x = keep_feature_local(x, self.k)
        # x = torch.cat((x, x0), dim=1)
        x = self.edge_cov4(x)
        x = torch.max(x, dim=2)[0]

        x = self.edge_cov5(x)
        x = torch.max(x, dim=2)[0]

        x = torch.cat([x1, x.unsqueeze(-1).repeat(1, 1, N)], dim=1)

        x = self.edge_cov6(x)
        x = self.edge_cov7(x)
        x = self.edge_cov8(x)
        return x
