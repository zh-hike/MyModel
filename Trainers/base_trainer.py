from Trainers.MyTrainer import MyTrainer


class CreateTrainer:
    def __init__(self, args, device):
        self.trainer = MyTrainer(args, device)

    def train_a_batch(self, views):
        targets = self.trainer.train_a_batch(views)
        return targets

    def save_model(self):
        self.trainer.save_model()

