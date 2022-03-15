from dataset.voc import Voc
from torch.utils.data import DataLoader


class Data:
    def __init__(self, args):
        self.args = args
        self.data = None
        self.select_data()
        self.dataloader = DataLoader(self.data, shuffle=True, batch_size=self.args.config['batch_size'])
        self.n_sample = self.data.n_sample
        self.dims = self.data.dims
        self.n_view = self.data.n_view

    def select_data(self):
        if self.args.dataset == 'voc':
            self.data = Voc(standard=self.args.config['standard_method'])

