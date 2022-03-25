import sys
import faiss
import torch
from loss.utils import SimilarityMatrix
import math
import torch.nn.functional as F
import torch.nn as nn
from loss.utils import get_neighbor
import torch
import numpy as np


class Dsim:
    """
    EAMC 论文中损失
    """

    def __init__(self, k, device):
        self.k = k
        print("k: ", k)
        self.device = device

    def __call__(self, A, attention_zs):
        K = SimilarityMatrix(attention_zs)
        e = torch.eye(self.k, self.k, device=self.device)
        beta = (A.unsqueeze(1) - e).abs().sum(dim=2)
        beta = torch.exp(beta)
        v = beta.T @ K @ beta
        diag = v.diag().sqrt()
        v = v / diag
        v = v / diag.unsqueeze(1)
        D = torch.triu(v, diagonal=1).sum() / self.k
        # print(D)
        return D


class Dsc:
    """
    EAMC
    论文中损失
    """

    def __init__(self, k):
        self.k = k

    def __call__(self, A, attention_zs):
        K = SimilarityMatrix(attention_zs)
        v = A.T @ K @ A
        diag = v.diag().sqrt()
        v = v / diag
        v = v / diag.unsqueeze(1)
        D = torch.triu(v, diagonal=1).sum() / self.k
        # print(D)
        return D


class CL:
    """
    Completer 损失
    """

    def __init__(self, alpha=9):
        self.alpha = alpha
        self.eps = sys.float_info.epsilon
        pass

    def __call__(self, zs):
        n_view = len(zs)
        # m = zs[0].shape[0]
        losses = 0

        for i in range(n_view):
            for j in range(i + 1, n_view):
                zi = zs[i].unsqueeze(2)
                zj = zs[j].unsqueeze(1)
                PP = zi @ zj

                PP = PP.mean(dim=0)
                PP = (PP + PP.t()) / 2
                PP = PP / PP.sum()
                PP_row = PP.sum(dim=1).detach()
                PP_col = PP.sum(dim=0).detach()
                PP[PP < self.eps] = self.eps
                PP_row[PP_row < self.eps] = self.eps
                PP_col[PP_col < self.eps] = self.eps
                # print(PP_row.shape, PP_col.shape)
                # mu = PP_row @ PP_col
                # mu[mu < self.eps] = self.eps
                # new_PP = PP / mu

                loss = -PP * (
                        torch.log(PP) - (self.alpha + 1) * torch.log(PP_row) - (self.alpha + 1) * torch.log(PP_col))
                # print(new_PP)
                # assert 1==0
                losses += loss
        return losses.sum()


class AGCLoss(nn.Module):
    def __init__(self, device, entropy_weight=2.0):
        super(AGCLoss, self).__init__()
        self.device = device
        self.xentropy = nn.CrossEntropyLoss().cuda()
        self.lamda = entropy_weight
        self.softmax = nn.Softmax(dim=1)
        self.temperature = 1.0

    def forward(self, attention_xs, pred):
        """
        :param attention_xs:  tensor x
        :param pred:       tensor the pred of x
        :return:
        """
        # print(attention_xs.device, pred.device)
        neighbor_index = get_neighbor(attention_xs)
        plogits = pred[neighbor_index]
        ologits = pred

        assert ologits.shape == plogits.shape, ('Inputs are required to have same shape')

        ologits = self.softmax(ologits)
        plogits = self.softmax(plogits)

        # one-hot
        similarity = torch.mm(F.normalize(ologits.t(), p=2, dim=1), F.normalize(plogits, p=2, dim=0))
        loss_ce = self.xentropy(similarity, torch.arange(similarity.size(0), device=self.device))

        # balance regularisation
        o = ologits.sum(0).view(-1)
        o /= o.sum()

        loss_ne = math.log(o.size(0)) + (o * o.log()).sum()

        loss = loss_ce + self.lamda * loss_ne

        # return loss, loss_ce, loss_ne
        return loss


class ProtoNCE:
    def __init__(self, device, alpha=9, clusters=[3, 5, 7]):
        self.device = device
        self.clusters = clusters
        self.alpha = alpha
        self.cri = nn.CrossEntropyLoss()

    def __call__(self, z):
        # z = F.normalize(z, p=2)
        n, dim = z.shape
        losses = []
        # data = {'Centers': [], 'phis': []}
        dz = z.detach().cpu().numpy()
        for n_cluster in self.clusters:
            clus = faiss.Clustering(dim, n_cluster)
            res = faiss.StandardGpuResources()
            index = faiss.IndexFlatL2(dim)
            index = faiss.index_cpu_to_gpu(res, 0, index)
            clus.train(dz, index)
            D, I = index.search(dz, 1)
            D = D.squeeze()
            I = I.squeeze()
            centers = faiss.vector_to_array(clus.centroids)
            DCluster = [[] for i in range(n_cluster)]
            for index, d in enumerate(D):
                DCluster[I[index]].append(d)

            phis = []
            for i in DCluster:
                phi = (np.array(i) ** 0.5).mean() / np.log(len(i) + self.alpha)
                phis.append(phi)
            phis = np.array(phis)
            phi_max = phi.max()
            for i, DC in enumerate(DCluster):
                if len(DC) <= 1:
                    phis[i] = phi_max
            phis = phis.clip(np.percentile(phis, 10), np.percentile(phis, 90))

            phis = torch.from_numpy(phis).to(self.device)
            centers = torch.tensor(centers, device=self.device).view(n_cluster, -1)
            # print(['%.2f'%i for i in phis])

            centers = nn.functional.normalize(centers, p=2, dim=1)
            sims = (z @ centers.T) / phis
            # print(sims.shape)
            # print(I.shape, I.min(), I.max())
            I = torch.tensor(I, dtype=torch.int64, device=self.device)
            sims = self.cri(sims, I)


            # sims = torch.exp(sims)
            # sims = sims / sims.sum(dim=1).unsqueeze(1)
            # sims = -1*torch.log(sims).mean()
            losses.append(sims)
        loss = (sum(losses) / len(losses))
        # print(loss)
        return loss
