import torch


def GaussianMatrix(z, sigma=1):
    """
    计算高斯度量矩阵
    :param z:
    :return:
    """
    z = z - z.unsqueeze(dim=1)
    K = torch.sqrt((z**2).sum(dim=2))
    K = -1*K/(2*(sigma**2))
    K = torch.exp(K)

    return K


def SimilarityMatrix(z):

    z = z / torch.sqrt((z ** 2).sum(dim=1) + 1e-5).unsqueeze(1)

    z = z @ z.T

    return z.abs()