import torch
from models.MyModel import AutoEncoder
from loss.OtherLoss import MSELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau


class MyTrainer:
    def __init__(self, args, device):
        self.args = args
        self.dataset = self.args.dataset
        self.pretrain = False
        self.epochs = self.args.config['epochs']
        if self.args.pretrain and self.args.config['needpretrain']:
            self.pretrain = True
            self.epochs = self.args.config['pre_epochs']

        self.device = device
        self._init_model()

    def _init_model(self):
        """
        初始化一下网络，优化器，损失函数等等，涉及到预训练的网络与正式训练
        :return:
        """
        self.autoencoder = AutoEncoder(self.args)
        print(self.autoencoder)
        self.autoencoder.to(self.device)
        self.auto_opt = torch.optim.Adam(self.autoencoder.parameters(),
                                         lr=self.args.config['network'][self.dataset]['autoencoder']['lr']
                                         )
        self.auto_sche = ReduceLROnPlateau(self.auto_opt,
                                           patience=self.epochs / 5,
                                           factor=0.5,
                                           verbose=True, )
        self.mseloss = MSELoss()

    def train_a_batch(self, views):
        """
        训练一个batch，
        :param views:
        :return: labels ，预测结果
        """
        self.loss = 0
        self.zs, self.xs_bar = self.autoencoder(views)
        autoloss = 0
        for x, x_bar in zip(views, self.xs_bar):
            autoloss += self.mseloss(x, x_bar)

        self.loss += autoloss

        if not self.pretrain:
            self.add_compare_loss()

        self._grad_zero()
        self._backward()
        self._step()

        if self.pretrain:
            return None
        else:
            return -1

    def add_compare_loss(self):

        pass

    def _backward(self):
        """
        反向传播，
        :return:
        """
        self.loss.backward()

    def _step(self):
        """
        优化器更新
        :return:
        """
        self.auto_opt.step()
        self.auto_sche.step(self.loss.item())

    def _grad_zero(self):
        """
        梯度清零
        :return:
        """
        self.auto_opt.zero_grad()

    def save_model(self):
        auto_statedict = self.autoencoder.state_dict()
        auto_opt = self.auto_opt.state_dict()
        if self.pretrain:
            data = {'net': auto_statedict,
                    'opt': auto_opt, }
            torch.save(data, './results/%s/StateDict/preTrainNet.pt' % self.args.model)
