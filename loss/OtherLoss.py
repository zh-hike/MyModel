import torch.nn as nn


class MSELoss:
    def __init__(self):
        self.cri = nn.MSELoss()

    def __call__(self, pred, real):
        return self.cri(pred, real)
