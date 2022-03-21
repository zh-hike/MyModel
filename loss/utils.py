import torch

def get_neighbor(x):
    x = x / x.norm(dim=1).unsqueeze(1)
    d = x.mm(x.T)
    n = x.shape[0]
    dig_index = list(range(n))
    d[dig_index, dig_index] = 0
    neighbor_index = d.argmax(dim=1)

    return neighbor_index


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