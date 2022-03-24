from dataset.voc import Voc
from torch.utils.data import DataLoader
from dataset.mnist import Mnist
from dataset.Caltech import Caltech

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
        print(self.args.dataset)
        if self.args.dataset == 'voc':

            self.data = Voc(standard=self.args.config['network'][self.args.dataset]['standard_method'])
        elif self.args.dataset == 'mnist':

            self.data = Mnist(standard=self.args.config['network'][self.args.dataset]['standard_method'])

        elif self.args.dataset == 'Caltech':

            self.data = Caltech(standard=self.args.config['network'][self.args.dataset]['standard_method'])
