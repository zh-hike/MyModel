import torch.nn as nn
import torch
from loss.utils import GaussianMatrix, SimilarityMatrix


class MSELoss:
    def __init__(self):
        self.cri = nn.MSELoss()

    def __call__(self, pred, real):
        return self.cri(pred, real)


class Dreg:
    """
    reg损失，对于聚类层的输出结果，让聚类结果的相关性矩阵上三角元素尽量为0
    """

    def __init__(self):
        pass

    def __call__(self, A):
        d = A.T @ A
        return torch.triu(d, diagonal=1).sum()


class AttLoss:
    def __init__(self, sigma=1):
        self.sigma = sigma
        pass

    def __call__(self, zs, ws, attention_zs):
        Kc = 0
        # Kf = GaussianMatrix(attention_zs, self.sigma)
        Kf = SimilarityMatrix(attention_zs)
        for z, w in zip(zs, ws):
            K = SimilarityMatrix(z)
            Kc += w * K


        K_loss = torch.sqrt(((Kc - Kf)**2).sum())
        # print(K_loss)
        return K_loss