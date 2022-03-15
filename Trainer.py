import torch
from dataset.dataLoader import Data
from Trainers.base_trainer import CreateTrainer
"""
Trainer.py 的作用是做一个整体的训练框架，
读取数据，加载调用models.base_model里
来选择通用模型，这个文件接受来自args的参数，
来训练模型，作为run.py或者参数实验文件和
模型训练（黑盒）之间的桥梁。
"""


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cpu')
        if self.args.which_gpu != -1:
            self.device = torch.device('cuda:%d' % self.args.which_gpu)
        self._init_data()

    def _init_data(self):
        self.data = Data(self.args)
        self.n_view = len(self.args.config['views_select'][self.args.dataset])
        self.n_sample = self.data.n_sample
        self.dims = [self.data.dims[i] for i in self.args.config['views_select'][self.args.dataset]]
        self.dataloader = self.data.dataloader

    def train(self):
        trainer = CreateTrainer(self.args)
        for epoch in range(1, self.args.config['epochs'] + 1):
            for batch, (views, targets) in enumerate(self.dataloader, 1):
                views = [views[i].to(self.device) for i in self.args.config['views_select'][self.args.dataset]]
                pred = trainer.train_a_batch(views)


