import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

class Dataset:
    def __init__(self, dataset_root_dir, batch_size, mode='train', point_num=5000, channel_num=3, size=493):
        self.mode = mode
        self.batch_size = batch_size
        self.dataset_root_dir = dataset_root_dir
        self.point_num = point_num
        self.channel_num = channel_num
        self.size = size

    def generate_downsample(self, down_sample_num=2048):
        # [B, N, C]
        data = np.zeros([self.size, down_sample_num, self.channel_num])
        # [B, N]
        label = np.zeros([self.size, down_sample_num])
        indices = np.random.choice(self.point_num, down_sample_num, replace=False)

        for i in range(self.size):
            data_path = self.dataset_root_dir + '/Data/' + self.mode + '/data_' + str(i + 1) + '.txt'
            label_path = self.dataset_root_dir + '/Label/' + self.mode + '/label_' + str(i + 1) + '.txt'
            data_tmp = np.loadtxt(data_path)
            label_tmp = np.loadtxt(label_path)
            data_tmp = data_tmp[indices, :]
            label_tmp = label_tmp[indices]
            data[i] = data_tmp
            label[i] = label_tmp

        dataset = TensorDataset(torch.tensor(data), torch.tensor(label))
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        return data_loader

    def generate_full_size(self):
        # [B, N, C]
        data = np.zeros([self.size, self.point_num, self.channel_num])
        # [B, N]
        label = np.zeros([self.size, self.point_num])

        for i in range(self.size):
            data_path = self.dataset_root_dir + '/Data/' + self.mode + '/data_' + str(i + 1) + '.txt'
            label_path = self.dataset_root_dir + '/Label/' + self.mode + '/label_' + str(i + 1) + '.txt'
            data_tmp = np.loadtxt(data_path)
            label_tmp = np.loadtxt(label_path)
            data[i] = data_tmp
            label[i] = label_tmp

        dataset = TensorDataset(torch.tensor(data), torch.tensor(label))
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        return data_loader
