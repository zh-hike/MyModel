import torch
from models.MyModel import AutoEncoder, Cluster
from loss.OtherLoss import MSELoss, Dreg, AttLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.cluster import KMeans
from models.utils import weight_init



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
        # print(self.autoencoder)
        self.autoencoder.to(self.device)
        self.auto_opt = torch.optim.Adam(self.autoencoder.parameters(),
                                         lr=self.args.config['network'][self.dataset]['autoencoder']['lr']
                                         )
        self.auto_sche = ReduceLROnPlateau(self.auto_opt,
                                           patience=self.epochs / 5,
                                           factor=0.5,
                                           verbose=True, )

        self.clusternet = Cluster(self.args)
        self.clusternet.to(self.device)
        self.cluster_opt = torch.optim.Adam(self.clusternet.parameters(),
                                            lr=self.args.config['network'][self.dataset]['cluster']['lr'])
        self.cluster_sche = ReduceLROnPlateau(self.cluster_opt,
                                              patience=self.epochs / 5,
                                              factor=0.5,
                                              verbose=True,
                                              )

        # self.autoencoder.apply(weight_init)
        self.clusternet.apply(weight_init)
        self.mseloss = MSELoss()
        self.dregloss = Dreg()
        self.att_loss = AttLoss(self.args.sigma)
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
        self.zs, self.attention_zs, self.xs_bar, self.ws = self.autoencoder(views)
        autoloss = 0
        for x, x_bar in zip(views, self.xs_bar):
            autoloss += self.mseloss(x, x_bar)

        self.loss += autoloss

        if not self.pretrain:
            pred = self.add_compare_loss()

        self._grad_zero()
        self._backward()
        self._step()

        if self.pretrain:
            # pred = self.cluster(self.attention_zs.detach().cpu().numpy())
            # return pred
            return None
        else:
            return pred.argmax(dim=1).detach().cpu().numpy()

    def cluster(self, data):
        model = KMeans(self.args.config['network'][self.dataset]['n_classes'])
        pred = model.fit_predict(data)
        return pred

    def add_compare_loss(self):
        pred = self.clusternet(self.attention_zs)

        self.loss += 2*self.dregloss(pred)

        # self.loss += 1*self.att_loss(self.zs, self.ws, self.attention_zs)
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
        auto_sche = self.auto_sche.state_dict()

        cluster_net = self.clusternet.state_dict()
        cluster_opt = self.cluster_opt.state_dict()
        cluster_sche = self.cluster_sche.state_dict()

        data = {'auto_net': auto_statedict,
                'auto_opt': auto_opt,
                'auto_sche': auto_sche,
                'cluster_net': cluster_net,
                'cluster_opt': cluster_opt,
                'cluster_sche': cluster_sche,
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
            self.clusternet.load_state_dict(data['cluster_net'])
            self.cluster_opt.load_state_dict(data['cluster_opt'])
            self.cluster_sche.load_state_dict(data['cluster_sche'])
        else:
            print("加载预训练模型参数")
            data = torch.load('./results/%s/StateDict/preTrainNet.pt' % self.args.model)

        self.autoencoder.load_state_dict(data['auto_net'])
        self.auto_opt.load_state_dict(data['auto_opt'])
        self.auto_sche.load_state_dict(data['auto_sche'])
