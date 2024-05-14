# 这个文件中主要是用来进行一些实验的
import torch
import numpy as np
import cv2
from skimage import exposure
from matplotlib import pyplot as plt
from tqdm import tqdm

from Dataset.RHD import RHDDataset
from Dataset.IsoGD import IsoGDDataset
from Segmentation import RDFSegmentor

def test_rhd_dataset_depth_range(dataset: RHDDataset):
    """
    测试RHD数据集的深度分布
    """
    row_num = 4
    col_num = 8
    fig, axes = plt.subplots(row_num, col_num, sharex=True, figsize=(20, 10))

    for i in range(row_num * col_num):
        # 随机选择几个样本
        sample_idx = np.random.randint(len(dataset))
        sample = dataset[sample_idx]
        depth = sample.depth.cpu().numpy()
        # 进行直方图均衡化
        # cv2.equalizeHist(depth, depth)
        equ_image = exposure.equalize_hist(depth)
        depth = equ_image.flatten()
        axe = axes[i // col_num, i % col_num]
        axe.hist(depth, bins=100)
        axe.set_title(f"样本 {sample_idx}\n背景像素数量: {len(depth[depth > 0.999])}")
    
    fig.suptitle("样本类别统计")
    plt.tight_layout()
    plt.show()

def test_rhd_dataset_mask_count(dataset: RHDDataset):
    """
    测试RHD数据集的类别分布
    """
    row_num = 2
    col_num = 4
    fig, axes = plt.subplots(row_num, col_num, sharex=True, figsize=(20, 10))

    for i in range(row_num * col_num):
        # 随机选择几个样本
        sample_idx = np.random.randint(len(dataset))
        sample = dataset[sample_idx]
        mask = sample.mask
        mask_count = np.bincount(mask.flatten())
        mask_len = 4
        # 将 mask_count 补充为 mask_len 个元素
        if mask_count.shape[0] < mask_len:
            mask_count = np.concatenate((mask_count, np.zeros(mask_len - mask_count.shape[0])))
        
        axe = axes[i // col_num, i % col_num]
        axe.bar(np.arange(mask_len), mask_count)
        axe.set_title(f"样本 {sample_idx}")
        axe.set_xticks(np.arange(mask_len))
        axe.set_xticklabels(['背景', '人体其它部分', '左手', '右手'])
    
    fig.suptitle("样本类别统计")
    plt.tight_layout()
    plt.show()

def test_rhd_dataset_depth_max(dataset: RHDDataset):
    """
    测试RHD数据集的深度最大值 （除背景外）
    """
    depth_max_list = []
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        depth = sample.depth.cpu().numpy()
        depth_max_list.append(depth[depth < 0.99].max())

    # 绘制直方图
    plt.hist(depth_max_list, bins=256)
    plt.title("RHD数据集样本深度最大值分布（除背景外）")
    plt.xlabel("深度最大值")
    plt.ylabel("样本数量")
    plt.show()


def test_iso_dataset_depth_range(dataset: IsoGDDataset, sample_idx: int = -1):
    """
    测试IsoGd数据集的深度分布
    """
    sample_idx = np.random.randint(len(dataset)) if sample_idx == -1 else sample_idx
    sample = dataset[sample_idx]
    row_num = 4
    col_num = 8
    fig, axes = plt.subplots(row_num, col_num, sharex=True, figsize=(20, 10))

    frame_count = min(len(sample), row_num * col_num)
    for i in range(frame_count):
        depth = sample.get_depth_frame(i)
        depth = depth.cpu().numpy().flatten()
        
        axe = axes[i // col_num, i % col_num]
        axe.hist(depth, bins=100)
        axe.set_title(f"第 {i + 1} 帧\n背景像素数量: {len(depth[depth > 0.999])}")
    
    fig.suptitle(f"样本 {sample_idx} 各帧深度分布")
    plt.tight_layout()
    plt.show()


def test_depth_feature(dataset: RHDDataset):
    """
    测试不同偏移向量下的深度特征直方图
    """
    sample_idx = np.random.randint(len(dataset))
    sample = dataset[sample_idx]
    row_num = 4
    col_num = 8
    uv_list = [np.random.random(4) * 20 - 10 for _ in range(row_num * col_num)]
    
    # 生成feature的直方图
    fig, axes = plt.subplots(row_num, col_num, sharex=True, figsize=(20, 10))
    for i in range(row_num * col_num):
        u = torch.tensor(uv_list[i][:2])
        v = torch.tensor(uv_list[i][2:])
        feature = RDFSegmentor.get_depth_feature(sample.depth, u, v).cpu()
        feature = feature.cpu().numpy().flatten()
        
        # 添加子图
        axe = axes[i//col_num, i%col_num]
        non_zero_count = (np.abs(feature) > 0.001).sum()
        # feature = feature.numpy()[np.where(sample.depth.cpu() < 0.999)].flatten()
        axe.set_title(f"u: ({u[0]:.2f}, {u[1]:.2f})\nv: ({v[0]:.2f}, {v[1]:.2f})\n非零像素数量: {non_zero_count}")
        axe.hist(feature, bins=100)
        # 设置坐标轴为log
        axe.set_yscale('log')

    fig.suptitle(f"不同偏移向量下的深度特征直方图 样本 {sample_idx}")
    plt.tight_layout()
    plt.show()

def test_depth_feature_mask(dataset: RHDDataset):
    """
    测试不同类别的深度特征直方图
    """
    sample_idx = np.random.randint(len(dataset))
    sample = dataset[sample_idx]
    u = torch.tensor((10, 10))
    v = torch.tensor((-7, 8))
    feature = RDFSegmentor.get_depth_feature(sample.depth, u, v)
    feature = feature.cpu().numpy().flatten()
    mask = sample.origin_mask.flatten()
    depth = sample.depth.cpu().numpy().flatten()
    
    # 生成feature的直方图
    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(10, 10))

    # 为每个类别生成直方图
    for i in range(4):
        # 根据 mask 来找到属于当前类别 i 的 depth 值
        depth_values = feature[np.where(mask == i)]
        # depth_values = feature[np.where((mask == i) & (depth < 0.99))]
        
        # 绘制直方图
        axes[i // 2, i % 2].hist(depth_values, bins=100, range=(-1, 1), alpha=0.75)
        axes[i // 2, i % 2].set_title(f'类别 {i} 像素数量: {len(depth_values)}')

        fig.suptitle(f"不同类别的深度特征直方图 样本 {sample_idx}")

    plt.tight_layout()
    plt.show()