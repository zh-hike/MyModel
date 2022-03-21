from dataset.voc import Voc
from torch.utils.data import DataLoader
from dataset.mnist import Mnist


class Data:
    def __init__(self, args):
        self.args = args
        self.data = None

        self.select_data()
        self.dataset = self.args.dataset
        self.dataloader = DataLoader(self.data, shuffle=True, batch_size=self.args.config['network'][self.dataset]['batch_size'])
        self.n_sample = self.data.n_sample
        self.dims = self.data.dims
        self.n_view = self.data.n_view
        self.n_classes = self.data.n_classes

    def select_data(self):
        if self.args.dataset == 'voc':
            self.data = Voc(standard=self.args.config['network'][self.args.dataset]['standard_method'])
        elif self.args.dataset == 'mnist':
            print(self.args.dataset)
            self.data = Mnist(standard=self.args.config['network'][self.args.dataset]['standard_method'])
