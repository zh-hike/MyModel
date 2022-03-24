from config.modelConfig import my_model, Completer


class SetConfig:
    def __init__(self, args):
        if args.model == 'MyModel':
            print("loading MyModel config...")
            my_model(args)
        elif args.model == 'Completer':
            print("loading Completer config...")
            Completer(args)
