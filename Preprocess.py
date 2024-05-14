# 此文件用于进行预处理任务
import os
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt

from Config import Config
from Dataset.RHD import RHDDataset

def modify_rhd_depth():
    """
    修改RHD数据集的深度信息 并保存到新的文件中
    """
    # 加载数据集
    config = Config("rhd")
    dataset = RHDDataset(config.dataset_root, set_type=config.dataset_split, to_tensor=False)

    def modify_depth(depth: np.ndarray):
        bg_mask = depth > 0.99
        bg_count = bg_mask.sum()
        threshold = depth[~bg_mask].max()
        max_th = 1.0 - (1.0 - threshold) * np.random.uniform(0.0, 0.4)
        min_th = threshold + (1.0 - threshold) * np.random.uniform(0.0, 0.4)

        # 生成高斯分布的值
        loc = (max_th + min_th) / 2
        scale = (max_th - min_th) / 6
        normal_depth = np.random.normal(loc, scale, size=bg_count)

        # 添加噪音
        noise_scale = np.random.normal(1.0, 0.1, size=bg_count)
        normal_depth = normal_depth * noise_scale

        # 裁剪边界
        normal_depth[normal_depth < threshold] = loc
        normal_depth[normal_depth > 1.0] = loc

        # 修改原深度图并返回
        depth[bg_mask] = normal_depth
        return depth

    des = os.path.join(dataset.data_root, 'modified_depth')
    if not os.path.exists(des):
        os.makedirs(des)

    # 遍历数据集并修改深度信息
    for i in tqdm(range(len(dataset))):
        img_path = os.path.join(des, f'{i:05d}.png')
        if os.path.exists(img_path):
            continue
        sample = dataset[i]
        modified_depth = modify_depth(sample.depth)
        cv2.imwrite(img_path, modified_depth * 255, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    # 测试效果
    # sample_idx = np.random.randint(len(dataset))
    # sample = dataset[sample_idx]
    # plt.subplot(2, 2, 1)
    # plt.imshow(sample.depth, cmap='gray')
    # plt.subplot(2, 2, 2)
    # plt.hist(sample.depth.flatten(), bins=256)

    # modified_depth = modify_depth(sample.depth)
    # plt.subplot(2, 2, 3)
    # plt.imshow(modified_depth, cmap='gray')
    # plt.subplot(2, 2, 4)
    # plt.hist(modified_depth.flatten(), bins=256)

    # plt.show()

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='rhd', help='要进行的预处理任务，目前仅支持rhd')
    args = parser.parse_args()

    if args.task == 'rhd':
        modify_rhd_depth()