# 这个文件中主要是用来进行一些实验的
import os
import time
import math
import pickle
from typing import List
import torch
import pandas as pd
import numpy as np
import cv2
from skimage import exposure
from sklearn.tree import export_text
from matplotlib import pyplot as plt
from tqdm import tqdm
import logging

from Dataset.RHD import RHDDataset, RHDDatasetItem
from Dataset.IsoGD import IsoGDDataset
from Segmentation import RDFSegmentor, DecisionTree
from Evaluation import SegmentationEvaluation
from Utils import Utils

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


def test_train_tree(dataset: RHDDataset, segmentor: RDFSegmentor, save: bool = True) -> DecisionTree:
    """
    测试训练一颗决策树
    """
    ret = segmentor.train_tree(dataset)
    tree = ret.sk_tree
    n_nodes = tree.tree_.node_count
    features = tree.tree_.feature

    # 打印每个节点的决策规则
    feature_names = [None] * ret.features.shape[0]
    for i in range(n_nodes):
        if features[i] != -2:  # 如果不是叶节点
            f = ret.features[features[i]]
            name = f"u:{f[0]:.2f},v:{f[1]:.2f}"
            feature_names[features[i]] = name
    logging.info('\n' + export_text(tree, feature_names=feature_names, decimals=4))

    # 保存决策树
    if save:
        if not os.path.exists("SegModel/Test"):
            os.makedirs("SegModel/Test")
        pickle.dump(ret, open("SegModel/Test/tree.pkl", "wb"))
    return ret
def test_train_forest(dataset: RHDDataset, segmentor: RDFSegmentor, save: bool = True) -> DecisionTree:
    """
    测试训练随机森林
    """
    ret = segmentor.train_forest(dataset)

    # 保存随机森林
    if save:
        if not os.path.exists("SegModel/Test"):
            os.makedirs("SegModel/Test")
        pickle.dump(ret, open("SegModel/Test/forest.pkl", "wb"))
    return ret
def test_train_tree_iterative(dataset: RHDDataset, segmentor: RDFSegmentor, save: bool = True) -> DecisionTree:
    """
    使用迭代训练一颗决策树
    """
    ret = segmentor.train_tree_iterative(dataset)
    tree = ret.sk_tree
    n_nodes = tree.tree_.node_count
    features = tree.tree_.feature

    # 打印每个节点的决策规则
    feature_names = [None] * ret.features.shape[0]
    for i in range(n_nodes):
        if features[i] != -2:  # 如果不是叶节点
            f = ret.features[features[i]]
            name = f"u:{f[0]:.2f},v:{f[1]:.2f}"
            feature_names[features[i]] = name
    logging.info('\n' + export_text(tree, feature_names=feature_names, decimals=4))

    # 保存决策树
    if save:
        if not os.path.exists("SegModel/Test"):
            os.makedirs("SegModel/Test")
        pickle.dump(ret, open("SegModel/Test/tree_iterative.pkl", "wb"))
    return ret

def test_one_tree_predict(dataset: RHDDataset, segmentor: RDFSegmentor, tree: DecisionTree | None = None, sample_idx: int = -1, show_plt: bool = True) -> np.ndarray:
    """
    测试一颗决策树的预测效果
    :param tree: 决策树
    :param sample_idx: 样本编号
    :param show_plt: 是否显示结果图
    :return: 预测结果 (width, height)
    """
    sample_idx = np.random.randint(len(dataset)) if sample_idx == -1 else sample_idx
    sample = dataset[sample_idx]
    if tree is None:
        tree = pickle.load(open("SegModel/Test/tree.pkl", "rb"))

    # 预测结果
    pred = segmentor.predict_mask(sample.depth, tree)
    # 显示预测结果
    if show_plt:
        __show_segmentation_result(sample, pred)
    return pred
def test_forest_predict(dataset: RHDDataset, segmentor: RDFSegmentor, forest: List[DecisionTree] | None = None):
    """
    测试随机森林的预测效果
    """
    sample_idx = np.random.randint(len(dataset))
    sample = dataset[sample_idx]
    if forest is None:
        forest = pickle.load(open("SegModel/Test/forest.pkl", "rb"))

    # 预测结果
    return_prob = True
    pred = segmentor.predict_mask(sample.depth, forest, return_prob=return_prob)
    # 显示预测结果
    __show_segmentation_result(sample, pred, ~return_prob)

def test_one_tree_result(dataset: RHDDataset, segmentor: RDFSegmentor, 
                         tree_count: int = 100, 
                         predict_count: int = 10, 
                         force_train: bool = True,
                         force_predict: bool = True,
                         show_plt: bool = True):
    """
    测试单棵决策树的结果 找到效果比较好的决策树 以及其特征
    :param tree_count: 决策树数量
    :param predict_count: 预测次数
    :param force_train: 是否强制重新训练决策树
    :param force_predict: 是否强制重新预测
    :param show_plt: 是否显示结果图
    """
    trees = []
    if not os.path.exists("SegModel/Test/SingleTree"):
        os.makedirs("SegModel/Test/SingleTree")

    # 训练决策树
    for i in tqdm(range(tree_count), desc="训练决策树"):
        if force_train or not os.path.exists(f"SegModel/Test/SingleTree/tree_{i}.pkl"):
            tree = segmentor.train_tree(dataset)
            pickle.dump(tree, open(f"SegModel/Test/SingleTree/tree_{i}.pkl", "wb"))
        else:
            tree = pickle.load(open(f"SegModel/Test/SingleTree/tree_{i}.pkl", "rb"))
        trees.append(tree)

    # 遍历SegModel/Test/SingleTree文件夹 查找result文件
    file_list = []
    for file_name in os.listdir("SegModel/Test/SingleTree"):
        if file_name.startswith("result_") and file_name.endswith(".csv"):
            file_list.append(file_name)
    
    # 进行验证
    iou_list = []
    force_predict |= force_train
    if force_predict or len(file_list) == 0:
        # 预测结果 并 记入文件
        result_file = open(f"SegModel/Test/SingleTree/result_{Utils.get_time_str()}.csv", "w")
        result_file.write(f"tree_idx,iou,{','.join([f'sample_{i}' for i in range(predict_count)])}\n")

        for i in range(len(trees)):
        # for i in tqdm(range(len(trees)), desc="统计预测结果"):
            # 随机采样
            sample_idx = [np.random.randint(len(dataset)) for _ in range(predict_count)]
            samples = [dataset[idx] for idx in sample_idx]
            gt_mask = [sample.mask for sample in samples]

            # 预测
            timer = time.perf_counter()
            depth_map = torch.stack([sample.depth for sample in samples])
            predict_mask = segmentor.predict_mask(depth_map, trees[i])
            logging.info(f'预测耗时: {time.perf_counter() - timer:.4f}s')
            timer = time.perf_counter()

            # 计算 IoU
            iou = SegmentationEvaluation.mean_iou_static(predict_mask, gt_mask)
            iou_list.append(iou)
            logging.info(f"第 {i} 棵决策树的平均 IoU: {iou:.4f}")
            result_file.write(f"{i},{iou:.6f},{','.join([str(s) for s in sample_idx])}\n")
        result_file.close()
    else:
        # 读取结果文件
        file_name = file_list.sort()[-1]
        df = pd.read_csv(f"SegModel/Test/SingleTree/{file_name}")
        iou_list = df['iou'].tolist()

    # 绘制结果图
    if show_plt:
        plt.bar(np.arange(len(iou_list)), iou_list)
        plt.title("单棵决策树的平均 IoU 统计")
        plt.xlabel("决策树编号")
        plt.ylabel("IoU")

        plt.show()
def test_vary_max_depth(dataset: RHDDataset, segmentor: RDFSegmentor):
    """
    测试不同最大深度下的效果
    """
    max_depth_list = [i for i in range(1, 23)]
    # max_depth_list = [1, 3, 5, 10, 15, 20, 25, 30, 40, 50]
    result_list = []
    sample_idx = 0
    sample = dataset[sample_idx]

    for i in tqdm(range(len(max_depth_list)), desc="测试不同最大深度"):
        segmentor.max_depth = max_depth_list[i]
        tree = segmentor.train_tree(dataset, sample_indices=[sample_idx])
        result = test_one_tree_predict(dataset, segmentor, tree, sample_idx=sample_idx, show_plt=False)
        result_list.append(result)

    pic_count = 2 + len(result_list)
    row_count = math.ceil(math.sqrt(pic_count // 2))
    col_count = math.ceil(pic_count / row_count)
    fig, axes = plt.subplots(row_count, col_count, figsize=(20, 10))

    axes[0, 0].imshow(sample.depth.cpu(), cmap='gray')
    axes[0, 0].set_title("原始图像")
    axes[0, 1].imshow(sample.mask, cmap='gray')
    axes[0, 1].set_title("Ground Truth")
    for i in range(2, len(result_list) + 2):
        row_idx = i // col_count
        col_idx = i % col_count
        axe = axes[row_idx, col_idx]
        axe.imshow(result_list[i - 2], cmap='gray')
        axe.set_title(f"深度 {max_depth_list[i - 2]} IoU: {SegmentationEvaluation.mean_iou_static(result_list[i - 2], sample.mask):.4f}")
    
    fig.suptitle(f"不同最大深度下的结果 样本 {sample_idx}")
    plt.tight_layout()
    plt.show()
    

def __show_segmentation_result(sample: RHDDatasetItem, pred: np.ndarray, binary: bool = True):
    """
    显示分割结果
    :param sample: 样本数据
    :param pred: 预测结果 (width, height)
    """
    mask_color = np.zeros((sample.mask.shape[0], sample.mask.shape[1], 3), dtype=np.uint8)
    mask_color[sample.mask == 0] = [255, 255, 255]  # 背景
    mask_color[sample.mask == 1] = [0, 255, 0]  # 手部

    if binary:
        pred_color = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        pred_color[pred == 0] = [255, 255, 255]  # 背景
        pred_color[pred == 1] = [0, 255, 0]  # 手部
    else:
        pred_color = pred / 255.0

    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    axes[0].imshow(sample.depth.cpu(), cmap='gray')
    axes[0].set_title("原始图像")
    axes[1].imshow(mask_color)
    axes[1].set_title("Ground Truth")
    axes[2].imshow(pred_color, cmap='gray' if not binary else None)
    axes[2].set_title("预测结果")

    plt.tight_layout()
    plt.show()