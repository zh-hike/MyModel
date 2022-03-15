import torch


class MyTrainer:
    def __init__(self, args):
        self.args = args
        self._init_model()

    def _init_model(self):
        """
        初始化一下网络，优化器，损失函数等等，涉及到预训练的网络与正式训练
        :return:
        """
        pass

    def train_a_batch(self, views):
        """
        训练一个batch，
        :param views:
        :return: labels ，预测结果
        """
        pass

    def _backward(self):
        """
        反向传播，
        :return:
        """
        pass

    def _step(self):
        """
        优化器更新
        :return:
        """

        pass

    def _grad_zero(self):
        """
        梯度清零
        :return:
        """

        pass


