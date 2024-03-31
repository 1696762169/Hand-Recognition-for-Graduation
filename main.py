import os
import numpy as np
import cv2
import torch
import argparse
import matplotlib.pyplot as plt

from Config import Config
from PSO import PSOSolver
from Utils import Utils
from HandModel import HandModel
from Evaluation import Evaluation
from Segmentation.Segmentaion import SkinColorSegmentation
from Dataset.RHD import RHDDataset

def test_mask(dataset: RHDDataset):
    sample = dataset[0]
    rgb = cv2.cvtColor(sample.color, cv2.COLOR_YUV2RGB)
    person = np.zeros_like(sample.mask)
    person[sample.mask == 1] = 1
    hand = np.zeros_like(sample.mask)
    hand[sample.mask >= 2] = 1

    plt.figure()

    plt.subplot(1, 3, 1)
    plt.imshow(rgb)
    plt.axis('off')
    plt.title('RGB')

    plt.subplot(1, 3, 2)
    plt.imshow(person)
    plt.axis('off')
    plt.title('Person')

    plt.subplot(1, 3, 3)
    plt.imshow(hand)
    plt.axis('off')
    plt.title('Hand')

    plt.show()

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
    sample =dataset[0]
    # test_mask(dataset)

    seg = SkinColorSegmentation(dataset, config)
    seg.load_trained_model(-1, True)

    # particle_num = 10
    # dim_num = 27
    # bounds = HandModel.get_bounds()
    # models = [HandModel() for _ in range(particle_num)]
    # evaluation = Evaluation(config, dataset, models)
    # def evaluate(tensor: torch.Tensor) -> torch.Tensor:
    #     return evaluation.evaluate(tensor)
    # pso = PSOSolver(particle_num, dim_num, bounds, evaluate)
    # result = pso.solve()
    # print("result:", result)

    # model = HandModel()
    # # model.show()
    # # model.show(True)
    # picture = model.render()
    # print(picture.shape)
    # print(picture.dtype)
    # picture = picture[:, :, 0] / 256.0
    # plt.imshow(picture, cmap='gray')
    # plt.axis('off')
    # plt.show()
