import argparse

parse = argparse.ArgumentParser('PaperCode')
parse.add_argument('--model', type=str, choices=['hike', 'completer', 'EAMC'], default='hike')

parse.add_argument('--config', help="模型的配置参数")
args = parse.parse_args()

if __name__ == "__main__":
    pass
