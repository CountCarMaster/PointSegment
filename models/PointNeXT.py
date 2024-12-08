import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.utils.PointNeXTUtils import PointNetSetAbstraction, PointNetFeaturePropagation, InvResMLP

class PointNeXT(nn.Module):
    def __init__(self, num_classes):
        super(PointNeXT, self).__init__()
        self.sa1 = PointNetSetAbstraction(0.2, 32, 1024, 3 + 3, [32, 32, 64], False)
        self.ir1 = InvResMLP(0.2, 32, 64 + 3, [64, 256, 64], False)
        self.sa2 = PointNetSetAbstraction(0.4, 32, 256, 64 + 3, [64, 64, 128], False)
        self.ir2 = InvResMLP(0.4, 32, 128 + 3, [128, 512, 128], False)
        self.sa3 = PointNetSetAbstraction(0.8, 32, 64, 128 + 3, [128, 128, 256], False)
        self.ir3 = InvResMLP(0.8, 32, 256 + 3, [256, 1024, 256], False)
        self.sa4 = PointNetSetAbstraction(1.6, 32, 16, 256 + 3, [256, 256, 512], False)
        self.ir4 = InvResMLP(1.6, 16, 512 + 3, [512, 1024, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]
        l0_xyz = l0_xyz.transpose(1, 2)
        l0_points = l0_points.transpose(1, 2)
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l1_xyz, l1_points = self.ir1(l1_xyz, l1_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l2_xyz, l2_points = self.ir2(l2_xyz, l2_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l3_xyz, l3_points = self.ir3(l3_xyz, l3_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        l4_xyz, l4_points = self.ir4(l4_xyz, l4_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points.transpose(1, 2))
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points.transpose(1, 2))
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points.transpose(1, 2))

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        # x = x.permute(0, 2, 1)
        return x