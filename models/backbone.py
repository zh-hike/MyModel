import torch.nn as nn


class BackBone(nn.Module):
    """
    骨干网络，用法，
    layers = [100,50,50,50,100]
    batchnorm = Flase,
    activate = 'ReLU' or 'LeakyReLU'
    out_activate = 'ReLU' or 'LeakyReLU' or 'Sigmoid'
    """
    def __init__(self, layers, batchnorm=False, activate='ReLU', out_activate='ReLU', dropout=False):

        super(BackBone, self).__init__()
        at = nn.Identity()
        out_at = nn.Identity()
        dp = nn.Identity()
        if dropout:
            dp = nn.Dropout(0.1)

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
            net.append(dp)
            if batchnorm:
                net.append(nn.BatchNorm1d(layers[i]))
            net.append(at)

        net = net[:-1]
        net.append(out_at)
        self.net = nn.Sequential(*net)

    def forward(self, x):

        x = self.net(x)

        return x
