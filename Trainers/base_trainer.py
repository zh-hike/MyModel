from Trainers.MyTrainer import MyTrainer


class CreateTrainer:
    def __init__(self, args):
        self.trainer = MyTrainer(args)

    def train_a_batch(self, views):
        targets = self.trainer.train_a_batch()
        return targets
