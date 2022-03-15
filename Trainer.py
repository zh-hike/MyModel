import torch


"""
Trainer.py 的作用是做一个整体的训练框架，
读取数据，加载调用models.base_model里
来选择通用模型，这个文件接受来自args的参数，
来训练模型，作为run.py或者参数实验文件和
模型训练（黑盒）之间的桥梁。
"""

