import os
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
import argparse
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
import time

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
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='iso', help='config file name')
    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[
                            # logging.FileHandler(f"Log/log {Utils.get_time_str()}.txt"),
                            logging.StreamHandler()
                        ])
    

    config_file = str(args.config)
    config = Config(config_file)
    dataset = config.dataset

    # data_loader = DataLoader(dataset, batch_size=5, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn)
    batch_index = 0
    timer = time.perf_counter()
    # for batch in data_loader:
    #     logging.info(f"batch: {batch_index} time: {time.perf_counter() - timer}")
    #     batch_index += 1
    #     if batch_index > 50:
    #         break
    # for i in tqdm(range(len(dataset))):
    for i in range(len(dataset)):
        sample = dataset[i]
        if i > 250:
            break
    logging.info(f"time: {time.perf_counter() - timer}")
    
    # test_mask(dataset)

    # seg = SkinColorSegmentation(dataset, config)
    # seg.segment(sample)
    # seg_model = seg.load_trained_model(-1, True)
    # seg_model = seg.load_trained_model()
    # hist = torch.histc(seg_model.flatten(), bins=256, min=0.02, max=1)
    # plt.hist(hist.cpu().numpy(), bins=256)
    # plt.show()
    # print("{:.6f} {:.6f}".format(float(seg_model.max()), float(seg_model.min())))
    # print("大于种子阈值的颜色数量：", seg_model[seg_model >= config.seg_seed_threshold].numel())
    # print("大于区域阈值的颜色数量：", seg_model[seg_model >= config.seg_blob_threshold].numel())

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
