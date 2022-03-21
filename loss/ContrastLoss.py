import torch
from loss.utils import SimilarityMatrix
import math
import torch.nn.functional as F
import torch.nn as nn
from loss.utils import get_neighbor



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

    def __init__(self):
        pass

    def __call__(self, zs):
        n_view = len(zs)
        m = zs[0].shape[0]
        P = 0
        for i in range(m):
            for v1 in range(n_view):
                for v2 in range(v1+1, n_view):
                    P = P + zs[v1][i]


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