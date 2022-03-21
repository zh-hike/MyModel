import torch
from dataset.dataLoader import Data
from Trainers.base_trainer import CreateTrainer
from utils import eva
import numpy as np
from utils import get_missmatrix


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
        self.missing_rate = self.args.missing_rate
        self.pretrain = False
        self.epochs = self.args.config['epochs']
        if self.args.pretrain and self.args.config['needpretrain']:
            self.pretrain = True
            self.epochs = self.args.config['pre_epochs']
        self.device = torch.device('cpu')
        if self.args.which_gpu != -1:
            print("调整map_location:", self.args.which_gpu)
            self.device = torch.device('cuda:%d' % self.args.which_gpu)
        self._init_data()

    def _init_data(self):
        self.data = Data(self.args)
        self.n_view = len(self.args.config['views_select'][self.args.dataset])
        self.n_sample = self.data.n_sample
        self.dims = [self.data.dims[i] for i in self.args.config['views_select'][self.args.dataset]]
        self.dataloader = self.data.dataloader
        self.n_classes = self.data.n_classes

    def train(self):
        patten = 'epochs: %d/%d  [==============]  loss: %.8f      acc: %.4f     nmi:%.4f'
        trainer = CreateTrainer(self.args, self.device)
        for epoch in range(1, self.epochs + 1):
            all_pred = np.array([])
            all_target = np.array([])
            losses = 0
            for batch, (views, targets) in enumerate(self.dataloader, 1):
                views = [views[i].to(self.device) for i in self.args.config['views_select'][self.args.dataset]]
                pred = trainer.train_a_batch(views)
                loss = trainer.trainer.loss
                losses += loss.item()
                if pred is not None:
                    all_pred = np.concatenate([all_pred, pred.squeeze()])
                    all_target = np.concatenate([all_target, targets.numpy()])

            if self.pretrain:
                print("epochs: %s/%s  [=========]  loss: %.6f"%(epoch, self.epochs, losses))
            else:
                acc, nmi, ari, f1 = self.eva(all_pred, all_target)
                print(patten % (epoch, self.epochs, losses, acc, nmi))

            if self.args.eval:
                print("验证结束!!")
                break
        trainer.save_model()

    def eva(self, pred, targets):
        """
        评估
        :param pred:
        :param targets:
        :return:
        """
        # print(pred.shape)
        # print(targets.shape)
        acc, nmi, ari, f1 = eva(targets, pred, self.n_classes)
        return acc, nmi, ari, f1
