import os
import torch
import argparse
from Config import Config
from PSO import PSOSolver
from Utils import Utils
from HandModel import HandModel

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Utils.set_log_file(f"Log/log {Utils.get_time_str()}.txt")
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--config', type=str, default='test', help='config file name')
    # args = parser.parse_args()

    # config_file = str(args.config)
    # config = Config(config_file)
    # dataset = config.dataset

    # sample = dataset[0]

    # particle_num = 10
    # dim_num = 5
    # bounds = [(0, 1), (0, 2), (0, 100), (1, 2), (-1, 1)]
    # target = torch.tensor([(bound[1] - bound[0]) / 2 for bound in bounds]).float().cuda()
    # target = torch.tensor([bound[1] for bound in bounds]).float().cuda()

    # def evaluate(tensor: torch.Tensor) -> torch.Tensor:
    #     return (tensor - target).sum(dim=1)
    # pso = PSOSolver(particle_num, dim_num, bounds, evaluate)
    # result = pso.solve()
    # print("result:", result)

    model = HandModel()
    model.show()
        # model.test()
