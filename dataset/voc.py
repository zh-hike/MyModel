import numpy as np
from torch.utils.data import Dataset
from dataset.utils import Standard


class Voc(Dataset):
    def __init__(self, standard=None):
        data = np.load('./dataset/data/voc.npz')
        self.targets = data['labels']
        self.view_0, self.view_1 = data['view_0'], data['view_1']
        self.n_sample = 5649
        self.dims = [512, 399]
        self.n_view = 2
        self.convert_type(standard)

    def convert_type(self, standard):
        self.targets = self.targets.astype('int16')
        self.view_0 = self.view_0.astype('float32')
        self.view_1 = self.view_1.astype('float32')

    def __getitem__(self, item):
        return (self.view_0[item], self.view_1[item]), self.targets[item]

    def __len__(self):
        return self.n_sample
