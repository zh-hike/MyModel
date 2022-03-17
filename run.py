import argparse
from config.initConfig import SetConfig
from Trainer import Trainer
from utils import set_seed
parse = argparse.ArgumentParser('PaperCode')
parse.add_argument('--model', type=str, choices=['MyModel', 'Completer', 'EAMC'], default='MyModel')
parse.add_argument('--dataset', type=str, choices=['voc', ], default='voc')
parse.add_argument('--which_gpu', type=int, help="哪一块gpu，当为-1是，选择cpu", default=-1)
parse.add_argument('--eval', action='store_true', help="是否验证")
parse.add_argument('--pretrain', action='store_true', help="是否进行预训练，若模型本身不需要预训练，则忽略")
parse.add_argument('--purpose', type=str, choices=['train', 'ParamExperiment'], default='train')
parse.add_argument('--hidden_dim', type=int, help="隐藏层维度", default=32)
parse.add_argument('--sigma', type=float, help="高斯度量矩阵的超参数", default=2)
parse.add_argument('--config', help="模型的配置参数")

args = parse.parse_args()

SetConfig(args)

if __name__ == "__main__":
    set_seed(args.config['network'][args.dataset]['seed'])

    if args.purpose == 'train':
        trainer = Trainer(args)
        trainer.train()
