import os
import torch
import argparse
from Config import Config
from PSO import PSOSolver
from Utils import Utils
from HandModel import HandModel
from Evaluation import Evaluation
import matplotlib.pyplot as plt

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Utils.set_log_file(f"Log/log {Utils.get_time_str()}.txt")
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='test', help='config file name')
    args = parser.parse_args()

    config_file = str(args.config)
    config = Config(config_file)
    dataset = config.dataset

    particle_num = 10
    dim_num = 27
    bounds = HandModel.get_bounds()
    models = [HandModel() for _ in range(particle_num)]
    evaluation = Evaluation(dataset, models)
    def evaluate(tensor: torch.Tensor) -> torch.Tensor:
        return evaluation.evaluate(tensor)
    pso = PSOSolver(particle_num, dim_num, bounds, evaluate)
    result = pso.solve()
    print("result:", result)

    model = HandModel()
    # model.show()
    # model.show(True)
    picture = model.render()
    print(picture.shape)
    print(picture.dtype)
    picture = picture[:, :, 0] / 256.0
    plt.imshow(picture, cmap='gray')
    plt.axis('off')
    plt.show()
