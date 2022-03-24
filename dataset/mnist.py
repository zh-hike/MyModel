import numpy as np
from torch.utils.data import Dataset
from dataset.utils import Standard
import scipy.io as io
import torch


class Mnist(Dataset):
    def __init__(self, standard=None):
        data = io.loadmat('./dataset/data/mnist.mat')
        self.targets = data['truth'].squeeze()
        self.view_0, self.view_1 = data['X'][0][0].T, data['X'][0][1].T
        self.n_sample = 10000
        self.dims = [784, 256]
        self.n_view = 2
        self.n_classes = 10
        self.convert_type(standard)

    def convert_type(self, standard):
        self.targets = self.targets.astype('int')
        self.view_0 = torch.from_numpy(self.view_0.astype('float32'))
        self.view_1 = torch.from_numpy(self.view_1.astype('float32'))
        self.view_0 = Standard(self.view_0, standard)
        self.view_1 = Standard(self.view_1, standard)

    def __getitem__(self, item):
        return (self.view_0[item], self.view_1[item]), self.targets[item]

    def __len__(self):
        return self.n_sample
