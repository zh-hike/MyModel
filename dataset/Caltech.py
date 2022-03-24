import numpy as np
from torch.utils.data import Dataset
from dataset.utils import Standard
import scipy.io as io


class Caltech(Dataset):
    def __init__(self, standard=None):
        data = io.loadmat('./dataset/data/Caltech101-20.mat')
        self.targets = data['truth'].squeeze()
        self.view_0, self.view_1, self.view_2, self.view_3, self.view_4, self.view_5 = (data['X'][0][0].T,
                                                                                        data['X'][0][1].T,
                                                                                        data['X'][0][2].T,
                                                                                        data['X'][0][3].T,
                                                                                        data['X'][0][4].T,
                                                                                        data['X'][0][5].T,

                                                                                        )
        self.n_sample = 2386
        self.dims = [48, 40, 254, 1984, 512, 928]
        self.n_view = 6
        self.n_classes = 20
        self.convert_type(standard)

    def convert_type(self, standard):
        self.targets = self.targets.astype('int8')
        self.view_0 = self.view_0.astype('float32')
        self.view_1 = self.view_1.astype('float32')
        self.view_2 = self.view_2.astype('float32')
        self.view_3 = self.view_3.astype('float32')
        self.view_4 = self.view_4.astype('float32')
        self.view_5 = self.view_5.astype('float32')

        self.view_0 = Standard(self.view_0, standard)
        self.view_1 = Standard(self.view_1, standard)
        self.view_2 = Standard(self.view_2, standard)
        self.view_3 = Standard(self.view_3, standard)
        self.view_4 = Standard(self.view_4, standard)
        self.view_5 = Standard(self.view_5, standard)

    def __getitem__(self, item):
        return (self.view_0[item], self.view_1[item],self.view_2[item], self.view_3[item],self.view_4[item], self.view_5[item]),\
               self.targets[item]

    def __len__(self):
        return self.n_sample
