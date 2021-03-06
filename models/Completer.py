import torch
import torch.nn as nn
from models.backbone import Block, BackBone


class AutoEncoder(nn.Module):
    def __init__(self, args):
        super(AutoEncoder, self).__init__()
        dataset = args.dataset
        self.n_view = len(args.config['views_select'][dataset])
        view_select = args.config['views_select'][dataset]
        encs = [args.config['network'][dataset]['autoencoder']['encoder']['encs'][i] for i in view_select]
        hidden_dim = args.config['network'][dataset]['autoencoder']['hidden_dim']
        for i in range(self.n_view):
            encs[i].append(hidden_dim)
        decs = [args.config['network'][dataset]['autoencoder']['decoder']['decs'][i] for i in view_select]
        for i in range(self.n_view):
            decs[i] = [hidden_dim] + decs[i]

        enc_batchnorm = args.config['network'][dataset]['autoencoder']['encoder']['batchnorm']
        enc_activate = args.config['network'][dataset]['autoencoder']['encoder']['activate']
        enc_out_batchnorm = args.config['network'][dataset]['autoencoder']['encoder']['out_batchnorm']
        enc_out_activate = args.config['network'][dataset]['autoencoder']['encoder']['out_activate']
        dec_batchnorm = args.config['network'][dataset]['autoencoder']['decoder']['batchnorm']
        dec_activate = args.config['network'][dataset]['autoencoder']['decoder']['activate']
        dec_out_batchnorm = args.config['network'][dataset]['autoencoder']['decoder']['out_batchnorm']
        dec_out_activate = args.config['network'][dataset]['autoencoder']['decoder']['out_activate']

        self.encoder = Block(encs, enc_batchnorm, enc_activate, enc_out_activate, enc_out_batchnorm)
        self.decoder = Block(decs, dec_batchnorm, dec_activate, dec_out_activate, dec_out_batchnorm)

    def forward(self, views):

        zs = self.encoder(views)

        xs_bar = self.decoder(zs)
        return zs, xs_bar


class CrossAutoEncoder(nn.Module):
    def __init__(self, args):
        super(CrossAutoEncoder, self).__init__()
        dataset = args.dataset
        layer = args.config['network'][dataset]['crossAutoencoder']['layer']
        batchnorm = args.config['network'][dataset]['crossAutoencoder']['batchnorm']
        activate = args.config['network'][dataset]['crossAutoencoder']['activate']
        out_activate = args.config['network'][dataset]['crossAutoencoder']['out_activate']
        out_batchnorm = args.config['network'][dataset]['crossAutoencoder']['out_batchnorm']
        self.crossLayer1 = BackBone(layer, batchnorm, activate, out_activate, out_batchnorm)
        self.crossLayer2 = BackBone(layer, batchnorm, activate, out_activate, out_batchnorm)

    def forward(self, zs):
        z1, z2 = zs
        z2_bar, z1_bar = self.crossLayer1(z1), self.crossLayer2(z2)

        return z1_bar, z2_bar
