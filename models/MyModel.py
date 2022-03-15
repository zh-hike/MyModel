import torch
import torch.nn as nn
from models.backbone import BackBone


class Block(nn.Module):
    """
    多是图的BackBone组合
    dims = [[10,32,64..],
            [23,32,64..],
            ]

    """

    def __init__(self, dims, batchnorm=False, activate='ReLU', out_activate=None):
        super(Block, self).__init__()
        self.nets = nn.ModuleList([])
        for dim in dims:
            self.nets.append(BackBone(layers=dim, batchnorm=batchnorm, activate=activate, out_activate=out_activate))

    def forward(self, views):
        zs = []
        for x, net in zip(views, self.nets):
            zs.append(net(x))
        return zs


class Attention(nn.Module):
    def __init__(self, dims, tau, batchnorm=True, activate='LeakyReLU', out_activate='Sigmoid'):
        super(Attention, self).__init__()
        self.tau = tau
        self.net = BackBone(dims,
                            batchnorm=batchnorm,
                            activate=activate,
                            out_activate=out_activate,
                            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, views):
        x = torch.cat(views, dim=1)
        ws = self.net(x) / self.tau
        ws = self.softmax(ws)
        ws = ws.mean(dim=0)
        new_x = 0
        for i, w in enumerate(ws):
            new_x += views[i]*w
        return new_x


class AutoEncoder(nn.Module):
    def __init__(self, args):
        super(AutoEncoder, self).__init__()
        dataset = args.dataset
        self.n_view = len(args.config['views_select'][dataset])
        encs = args.config['network'][dataset]['autoencoder']['encoder']['encs']
        hidden_dim = args.config['network'][dataset]['autoencoder']['hidden_dim']
        for i in range(len(encs)):
            encs[i].append(hidden_dim)
        decs = args.config['network'][dataset]['autoencoder']['decoder']['decs']
        for i in range(len(decs)):
            decs[i] = [hidden_dim] + decs[i]

        enc_batchnorm = args.config['network'][dataset]['autoencoder']['encoder']['batchnorm']
        enc_activate = args.config['network'][dataset]['autoencoder']['encoder']['activate']
        enc_out_activate = args.config['network'][dataset]['autoencoder']['encoder']['out_activate']
        dec_batchnorm = args.config['network'][dataset]['autoencoder']['decoder']['batchnorm']
        dec_activate = args.config['network'][dataset]['autoencoder']['decoder']['activate']
        dec_out_activate = args.config['network'][dataset]['autoencoder']['decoder']['out_activate']

        attention_dim = args.config['network'][dataset]['attention']['dims']
        attention_batchnorm = args.config['network'][dataset]['attention']['batchnorm']
        attention_activate = args.config['network'][dataset]['attention']['activate']
        attention_out_activate = args.config['network'][dataset]['attention']['out_activate']
        tau = args.config['network'][dataset]['attention']['tau']
        self.encoder = Block(encs, enc_batchnorm, enc_activate, enc_out_activate)
        self.decoder = Block(decs, dec_batchnorm, dec_activate, dec_out_activate)
        self.attention = Attention(attention_dim,
                                   tau=tau,
                                   batchnorm=attention_batchnorm,
                                   activate=attention_activate,
                                   out_activate=attention_out_activate,
                                   )

    def forward(self, views):
        zs = self.encoder(views)
        zs = self.attention(zs)
        new_zs = [zs] * self.n_view
        xs_bar = self.decoder(new_zs)
        return zs, xs_bar
