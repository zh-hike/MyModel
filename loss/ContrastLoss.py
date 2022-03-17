import torch
from loss.utils import SimilarityMatrix


class Dsim:
    """
    EAMC 论文中损失
    """
    def __init__(self, k, device):
        self.k = k
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

