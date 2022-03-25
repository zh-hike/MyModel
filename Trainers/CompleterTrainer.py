import torch
from models.Completer import AutoEncoder, CrossAutoEncoder
from loss.OtherLoss import MSELoss, Dreg, AttLoss
from loss.ContrastLoss import Dsim, Dsc, AGCLoss, CL
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.cluster import KMeans
from models.utils import weight_init


class CompleterTrainer:
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
        # print(self.autoencoder)
        self.autoencoder.to(self.device)
        self.auto_opt = torch.optim.Adam(self.autoencoder.parameters(),
                                         lr=self.args.config['network'][self.dataset]['autoencoder']['lr']
                                         )
        self.auto_sche = ReduceLROnPlateau(self.auto_opt,
                                           patience=self.epochs / 5,
                                           factor=0.5,
                                           verbose=True, )
        self.crossAuto = CrossAutoEncoder(self.args)
        self.crossAuto.to(self.device)
        self.crossAuto_opt = torch.optim.Adam(self.crossAuto.parameters(),
                                              lr=self.args.config['network'][self.dataset]['crossAutoencoder']['lr'])
        self.crossAuto_sche = ReduceLROnPlateau(self.crossAuto_opt,
                                                patience=self.epochs / 5,
                                                factor=0.5,
                                                verbose=True, )
        # self.autoencoder.apply(weight_init)

        self.mseloss = MSELoss()
        self.dregloss = Dreg()
        self.att_loss = AttLoss(self.args.sigma)
        self.Dsim_loss = Dsim(self.args.config['network'][self.dataset]['n_classes'], self.device)
        self.Dsc_loss = Dsc(self.args.config['network'][self.dataset]['n_classes'])
        self.Agc_loss = AGCLoss(device=self.device)
        self.CL_loss = CL()
        print(self.crossAuto)
        print(self.autoencoder)

        # if self.args.eval:
        #     self.autoencoder.eval()
        #     self.clusternet.eval()

    def train_a_batch(self, views):
        """
        训练一个batch，
        :param views:
        :return: labels ，预测结果
        """
        self.loss = 0

        # zs: 多视图的list特征 ，  attention_zs: 进过attention融合后的，  xs_bar：自编码器的输出
        self.zs, self.xs_bar = self.autoencoder(views)
        autoloss = 0
        for x, x_bar in zip(views, self.xs_bar):
            autoloss += self.mseloss(x, x_bar)

        self.loss += 0.1 * autoloss

        self.loss += 1 * self.CL_loss(self.zs)

        if not self.pretrain:
            self.pred = self.add_compare_loss()

        self._grad_zero()
        self._backward()
        self._step()

        if self.pretrain:
            pred = self.cluster()
            return pred

        else:
            # return self.pred.argmax(dim=1).detach().cpu().numpy()
            return self.pred

    def cluster(self):
        model = KMeans(self.args.config['network'][self.dataset]['n_classes'])
        z = torch.cat(self.zs, dim=1).detach().cpu().numpy()
        pred = model.fit_predict(z)
        return pred

    def add_compare_loss(self):

        # self.loss += 0.1 * self.dregloss(pred)

        z1, z2 = self.zs
        z1_bar, z2_bar = self.crossAuto(self.zs)
        cross_loss = self.mseloss(z1_bar, z1) + self.mseloss(z2_bar, z2)
        self.loss += 0.1 * cross_loss

        # print(self.loss.device)
        # self.loss += 1*self.Agc_loss(self.attention_zs, pred)
        # self.loss += 0.01*self.att_loss(self.zs, self.ws, self.attention_zs)

        # self.loss += 100*self.Dsim_loss(pred, self.attention_zs)
        # self.loss += 1*self.Dsc_loss(pred, self.attention_zs)


        pred = self.cluster()

        return pred

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
        # self.auto_sche.step(self.loss.item())
        self.crossAuto_opt.step()
        # self.crossAuto_sche.step(self.loss.item())

    def _grad_zero(self):
        """
        梯度清零
        :return:
        """
        self.auto_opt.zero_grad()
        self.crossAuto_opt.zero_grad()

    def save_model(self):
        auto_statedict = self.autoencoder.state_dict()
        auto_opt = self.auto_opt.state_dict()
        auto_sche = self.auto_sche.state_dict()
        crossAuto_statedict = self.crossAuto.state_dict()
        cross_opt = self.crossAuto_opt.state_dict()
        cross_sche = self.crossAuto_sche.state_dict()

        data = {'auto_net': auto_statedict,
                'auto_opt': auto_opt,
                'auto_sche': auto_sche,
                'crossAuto_net': crossAuto_statedict,
                'cross_opt': cross_opt,
                'cross_sche': cross_sche,
                }
        if self.args.eval:
            return
        if self.pretrain:
            print("保存预训练模型参数")
            torch.save(data, './results/%s/StateDict/preTrainNet.pt' % self.args.model)
        else:
            print("保存模型参数")
            torch.save(data, './results/%s/StateDict/Net.pt' % self.args.model)

        # if self.pretrain:
        #     print("保存预训练参数")
        #     data = {'net': auto_statedict,
        #             'opt': auto_opt,
        #             'sche': auto_sche,
        #             }
        #     torch.save(data, './results/%s/StateDict/preTrainNet.pt' % self.args.model)
        # else:
        #     print("保存模型所有参数")

    def load_model(self):
        data = None
        if self.args.eval:
            print("加载模型参数")
            data = torch.load('./results/%s/StateDict/Net.pt' % self.args.model)

        else:
            print("加载预训练模型参数")
            data = torch.load('./results/%s/StateDict/preTrainNet.pt' % self.args.model)

        self.autoencoder.load_state_dict(data['auto_net'])
        self.auto_opt.load_state_dict(data['auto_opt'])
        self.auto_sche.load_state_dict(data['auto_sche'])
        self.crossAuto.load_state_dict(data['crossAuto_net'])
        self.crossAuto_opt.load_state_dict(data['cross_opt'])
        self.crossAuto_sche.load_state_dict(data['cross_sche'])
