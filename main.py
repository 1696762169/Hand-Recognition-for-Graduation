import os
import argparse
from Config import Config

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='test', help='config file name')
    args = parser.parse_args()

    config_file = str(args.config)
    config = Config(config_file)
    dataset = config.dataset

    sample = dataset[0]

    print(sample)
