# 此文件用于进行预处理任务
import os
import struct
from tqdm import tqdm
import argparse
import numpy as np
import cv2
import torch
from tqdm import tqdm
import logging
from matplotlib import pyplot as plt

from Config import Config
from Dataset.RHD import RHDDataset
from Dataset.IsoGD import IsoGDDataset
from Dataset.Senz import SenzDataset
from Segmentation import ResNetSegmentor, RDFSegmentor
from Tracker import ThreeDSCTracker
from Classification import DTWClassifier
from Evaluation import SegmentationEvaluation

def modify_rhd_depth():
    """
    修改RHD数据集的深度信息 并保存到新的文件中
    """
    # 加载数据集
    config = Config(config_path="data_RHD")
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
    dataset_config = Config(config_path="data_RHD")
    segment_config = Config(config_path="seg_RDF")
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
        predict_mask = segmentor.predict_mask_impl(depth_map, tree)
        test_iou = SegmentationEvaluation.mean_iou_static(predict_mask, gt_mask)

        # 筛选有效特征
        logging.info(f'Test IOU: {test_iou}')
        if test_iou < good_threshold:
            continue

        feature_indices = np.argsort(tree.sk_tree.feature_importances_)[::-1][:keep_feature_count]
        good_feature = tree.features[feature_indices.copy(), :].cpu().numpy()
        good_features = np.concatenate([good_features, good_feature], axis=0)

        if len(good_features) >= target_count:
            break

    # 保存有效特征
    np.save('SegModel/Test/good_features.npy', good_features)

def get_senz_list():
    """
    获取Senz3D数据集的文件列表
    """
    config = Config(config_path="data_Senz")
    dataset_root = config.dataset_root
    data_root = os.path.join(dataset_root, 'data')

    with open(os.path.join(dataset_root, 'data.txt'), 'w') as f:
        for sub_dir in os.listdir(data_root):
            sub_idx = int(sub_dir[1:])
            for ges_dir in os.listdir(os.path.join(data_root, sub_dir)):
                ges_idx = int(ges_dir[1:])
                img_idx = 1
                while os.path.exists(os.path.join(data_root, sub_dir, ges_dir, f'{img_idx}-color.png')):
                    f.write(f'{sub_dir}/{ges_dir} {sub_idx} {ges_idx} {img_idx}\n')
                    img_idx += 1

def split_iso_to_images():
    """
    将ISO数据集划分为图像文件
    """
    config = Config(config_path="data_IsoGD")
    dataset = IsoGDDataset(config.dataset_root, config.dataset_split, False, **config.get_dataset_params())
    sample = dataset[1]

    des = r'D:\Python\dataset\temp'
    for i in range(len(sample)):
        img_path = os.path.join(des, f'{i:05d}.png')
        cv2.imwrite(img_path, sample.get_rgb_frame(i), [cv2.IMWRITE_PNG_COMPRESSION, 0])

def calculate_features(dataset: RHDDataset | SenzDataset, segmentor: ResNetSegmentor, tracker: ThreeDSCTracker, output_dir: str):
    """
    获取特征向量并保存到文件中
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    no_label = isinstance(dataset, RHDDataset)
    file = open(os.path.join(output_dir, f'features_{type(dataset).__name__}_{dataset.set_type}.bin'), 'wb')
    
    seek_pos = []
    sample_count = len(dataset)
    file.seek(4 * (1 + sample_count))

    for i in tqdm(range(sample_count), desc='计算特征向量'):
        sample = dataset[i]
        pred = torch.tensor(sample.mask)
        depth_contour = tracker.extract_contour(sample.depth, pred)
        simplified_contour = tracker.simplify_contour(depth_contour)
        features = tracker.get_descriptor_features(simplified_contour, sample.depth)

        features = features.cpu().numpy().reshape(features.shape[0], -1)
        byte = DTWClassifier.to_byte(features, i, 0 if no_label else sample.label)
        seek_pos.append(file.tell())    # 记录文件指针位置

        # check_features, _, _ = DTWClassifier.from_byte(byte)
        file.write(struct.pack('I', len(byte)))
        file.write(byte)

    seek_pos = np.array(seek_pos, dtype=np.uint32)
    file.seek(0)
    file.write(struct.pack('I', len(seek_pos)))
    file.write(seek_pos.tobytes())
    file.close()

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='split_iso', help='要进行的预处理任务，目前仅支持rhd')
    args = parser.parse_args()

    task = args.task
    if task == 'rhd':
        modify_rhd_depth()
    elif task == 'good_features':
        get_good_features()
    elif task == 'senz':
        get_senz_list()
    elif task == 'split_iso':
        split_iso_to_images()
    else:
        print(f'暂不支持的预处理任务：{task}')