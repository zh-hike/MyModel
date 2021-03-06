import torch.nn as nn


class BackBone(nn.Module):
    """
    骨干网络，用法，
    layers = [100,50,50,50,100]
    batchnorm = Flase,
    activate = 'ReLU' or 'LeakyReLU'
    out_activate = 'ReLU' or 'LeakyReLU' or 'Sigmoid'
    """

    def __init__(self, layers, batchnorm=False, activate='ReLU', out_activate='ReLU', dropout=False, out_batchnorm=True):

        super(BackBone, self).__init__()
        at = nn.Identity()
        out_at = nn.Identity()
        # dp = nn.Identity()
        # if dropout:
        #     dp = nn.Dropout(0.1)

        if activate == 'ReLU':
            at = nn.ReLU(inplace=True)
        elif activate == 'LeakyReLU':
            at = nn.LeakyReLU(0.2, inplace=True)
        if out_activate == 'ReLU':
            out_at = nn.ReLU(inplace=True)
        elif out_activate == 'LeakyReLU':
            out_at = nn.LeakyReLU(0.2, inplace=True)
        elif out_activate == 'Sigmoid':
            out_at = nn.Sigmoid()
        elif out_activate == 'Softmax':
            out_at = nn.Softmax(dim=1)

        net = []
        for i in range(1, len(layers)):
            net.append(nn.Linear(layers[i - 1], layers[i]))
            # net.append(dp)
            if batchnorm:
                net.append(nn.BatchNorm1d(layers[i]))
            net.append(at)

        net = net[:-2]
        if out_activate is not None:
            if out_batchnorm:
                net.append(nn.BatchNorm1d(layers[i]))
            net.append(out_at)
        self.net = nn.Sequential(*net)

    def forward(self, x):

        x = self.net(x)

        return x


class Block(nn.Module):
    """
    多是图的BackBone组合
    dims = [[10,32,64..],
            [23,32,64..],
            ]

    """

    def __init__(self, dims, batchnorm=False, activate='ReLU', out_activate=None, out_batchnorm=True):
        super(Block, self).__init__()
        self.nets = nn.ModuleList([])
        for dim in dims:
            self.nets.append(
                BackBone(layers=dim,
                         batchnorm=batchnorm,
                         activate=activate,
                         out_activate=out_activate,
                         out_batchnorm=out_batchnorm
                         )
            )

    def forward(self, views):
        zs = []
        for x, net in zip(views, self.nets):
            zs.append(net(x))
        return zs
