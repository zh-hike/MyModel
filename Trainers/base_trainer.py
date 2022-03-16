from Trainers.MyTrainer import MyTrainer


class CreateTrainer:
    def __init__(self, args, device):
        self.trainer = MyTrainer(args, device)

        # 如果当前不想预训练，但是模型又需要预训练，则说明已经进行了预训练
        if (not args.pretrain) and args.config['needpretrain']:
            self.trainer.load_model()

    def train_a_batch(self, views):
        targets = self.trainer.train_a_batch(views)
        return targets

    def save_model(self):
        self.trainer.save_model()

