from config.modelConfig import my_model


class SetConfig:
    def __init__(self, args):
        if args.model == 'MyModel':
            my_model(args)

