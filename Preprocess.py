# 此文件用于进行预处理任务
import os
from tqdm import tqdm
import argparse
import numpy as np
import cv2
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt

from Config import Config
from Dataset.RHD import RHDDataset
from Segmentation import RDFSegmentor
from Evaluation import SegmentationEvaluation

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

def get_good_features():
    """
    获取较为有效的偏移向量
    """
    # 加载数据集和分类器
    dataset_config = Config("rhd")
    segment_config = Config("seg_rdf")
    dataset = RHDDataset(dataset_config.dataset_root, set_type=dataset_config.dataset_split, **dataset_config.get_dataset_params())
    segmentor = RDFSegmentor(**segment_config.get_seg_params())

    max_round = 10
    keep_feature_count = 10
    target_count = 2000

    test_count = 10
    good_threshold = 0.2
    good_features = np.empty((0, 4), dtype=np.float32)

    for _ in tqdm(range(max_round)):
        tree = segmentor.train_tree_iterative(dataset)

        # 进行评估
        test_indices = np.random.choice(len(dataset), size=test_count, replace=False)
        test_samples = [dataset[i] for i in test_indices]
        gt_mask = [sample.mask for sample in test_samples]
        
        depth_map = torch.stack([sample.depth for sample in test_samples])
        predict_mask = segmentor.predict_mask(depth_map, tree)
        test_iou = SegmentationEvaluation.mean_iou_static(predict_mask, gt_mask)

        # 筛选有效特征
        if test_iou < good_threshold:
            continue

        feature_indices = np.argsort(tree.sk_tree.feature_importances_)[::-1][:keep_feature_count]
        good_feature = tree.features[feature_indices.copy(), :].cpu().numpy()
        good_features = np.concatenate([good_features, good_feature], axis=0)

        if len(good_features) >= target_count:
            break

    # 保存有效特征
    np.save('SegModel/Test/good_features.npy', good_features)

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='good_features', help='要进行的预处理任务，目前仅支持rhd')
    args = parser.parse_args()

    task = args.task
    if task == 'rhd':
        modify_rhd_depth()
    elif task == 'good_features':
        get_good_features()
    else:
        print(f'暂不支持的预处理任务：{task}')