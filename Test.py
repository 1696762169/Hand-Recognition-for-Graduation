# 这个文件中主要是用来进行一些实验的
import os
import time
import math
import pickle
from typing import List, Tuple, Callable
import torch
from torchvision import transforms
import pandas as pd
import numpy as np
import cv2
from skimage import exposure
from fastdtw import fastdtw
from sklearn.tree import export_text
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm
import logging

from Config import Config
from Dataset.RHD import RHDDataset, RHDDatasetItem
from Dataset.IsoGD import IsoGDDataset
from Dataset.Senz import SenzDataset, SenzDatasetItem
from Segmentation import RDFSegmentor, DecisionTree
from Tracker import ThreeDSCTracker, LBPTracker
from Classification import DTWClassifier, FastDTW
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


def test_senz_dataset():
    """
    测试Senz数据集导入效果
    """ 
    from main import create_dataset
    dataset: SenzDataset = create_dataset(Config(dataset_type='Senz'))
    dataset.filter_type = 'none'
    dataset.to_tensor = False

    sample_idx = np.random.randint(len(dataset))
    sample = dataset[sample_idx]

    fig, axes = plt.subplots(3, 4, figsize=(20, 8))
    fig.suptitle(f"Senz数据集样本 {sample_idx}")

    axes[0, 0].imshow(sample.color_rgb)
    axes[0, 0].set_title("RGB")
    axes[1, 0].imshow(sample.depth, cmap='gray')
    axes[1, 0].set_title("Depth")
    axes[2, 0].imshow(sample.confidence, cmap='gray')
    axes[2, 0].set_title("Confidence")

    thresholds = np.arange(1, 10, 1) / 800
    for i, threshold in enumerate(thresholds):
        mask = sample.confidence > threshold
        axe = axes[i // 3, (i % 3) + 1]
        axe.imshow(mask, cmap='gray')
        axe.set_title(f"Mask of {threshold:.4f}")

    plt.tight_layout()
    plt.show()

def test_senz_preprocessing():
    """
    测试Senz数据集的滤波预处理效果
    """
    from main import create_dataset
    dataset: SenzDataset = create_dataset(Config(dataset_type='Senz'))
    dataset.filter_type = 'none'
    dataset.to_tensor = False
    
    sample_idx = np.random.randint(len(dataset))
    sample_idx = 1057
    sample = dataset[sample_idx]

    show_params = True
    show_confidence = False
    config_list = [
        ('gaussian', 5, 1),
        ('median', 5),
        ('bilateral', 5, 1, 1),

        # ('bilateral', 5, 0, 0),
        # ('bilateral', 5, 1, 1),
        # ('bilateral', 5, 2, 2),
        # ('bilateral', 5, 3, 3),
        # ('bilateral', 5, 4, 4),
        # ('bilateral', 5, 16, 16),
        # ('bilateral', 5, 64, 64),
        # ('bilateral', 5, 128, 128),

        # ('gaussian', 5, 1),
        # ('gaussian', 5, 4),
        # ('gaussian', 5, 16),
        # ('gaussian', 5, 64),
    ]

    row_num, col_num = 1, len(config_list) + (2 if show_confidence else 1)
    temp = plt.subplots(row_num, col_num, figsize=(20, 8))
    fig: plt.Figure = temp[0]
    axes: List[plt.Axes] = temp[1]
    for axe in axes:
        axe.set_axis_off()
    fig.suptitle(f"Senz数据集样本 {sample_idx} 预处理效果对比")
    title_y = -0.1

    # 原始图像
    axes[0].imshow(sample.depth, cmap='gray')
    axes[0].set_title("(a) 原始深度图", y=title_y, verticalalignment='top', fontsize=20)
    # 置信度图像
    if show_confidence:
        confidence = sample.confidence
        axes[1].imshow(confidence, cmap='jet')
        axes[1].set_title("(b) 手部置信度", y=title_y, verticalalignment='top', fontsize=20)

    for i in range(len(config_list)):
        axe_idx = i + 2 if show_confidence else i + 1
        axe = axes[axe_idx]
        config = config_list[i]
        if config[0] == 'bilateral':
            depth = cv2.bilateralFilter(sample.depth, d=config[1], sigmaColor=config[2], sigmaSpace=config[3])
            # depth = cv2.medianBlur(depth, 5)
            name = f"双边滤波"
            params = f"核大小: {config[1]}\n强度标准差: {config[2]}\n空间标准差: {config[3]}"
        elif config[0] == 'gaussian':
            depth = cv2.GaussianBlur(sample.depth, ksize=(config[1], config[1]), sigmaX=config[2])
            # depth = cv2.medianBlur(depth, 5)
            name = f"高斯滤波"
            params = f"核大小: {config[1]}\n标准差: {config[2]}"
        elif config[0] =='median':
            depth = cv2.medianBlur(sample.depth, config[1])
            # depth = cv2.bilateralFilter(depth, d=5, sigmaColor=8, sigmaSpace=8)
            # depth = cv2.medianBlur(depth, config[1])
            name = f"中值滤波"
            params = f"核大小: {config[1]}"
        else:
            raise ValueError(f"未知的滤波类型: {config}")
        axe.imshow(depth, cmap='gray')
        title = f"({chr(ord('a') + axe_idx)}) {params}" if show_params else f"({chr(ord('a') + axe_idx)}) {name}"
        axe.set_title(title, y=title_y, verticalalignment='top', fontsize=20)

    plt.tight_layout()
    plt.show()
def test_senz_combine_filter():
    """
    测试Senz数据集的组合滤波效果
    """
    from main import create_dataset
    dataset: SenzDataset = create_dataset(Config(dataset_type='Senz'))
    dataset.filter_type = 'none'
    dataset.to_tensor = False
    
    sample_idx = np.random.randint(len(dataset))
    sample_idx = 1057
    sample = dataset[sample_idx]

    row_num, col_num = 1, 5
    temp = plt.subplots(row_num, col_num, figsize=(20, 8))
    fig: plt.Figure = temp[0]
    axes: List[plt.Axes] = temp[1]
    for axe in axes:
        axe.set_axis_off()
    fig.suptitle(f"Senz数据集样本 {sample_idx} 预处理效果对比")
    title_y = -0.1

    # 原始图像
    axes[0].imshow(sample.depth, cmap='gray')
    axes[0].set_title("(a) 原始深度图", y=title_y, verticalalignment='top', fontsize=20)
    # 高斯滤波
    depth = cv2.GaussianBlur(sample.depth, ksize=(5, 5), sigmaX=1)
    axes[1].imshow(depth, cmap='gray')
    axes[1].set_title("(b) 高斯滤波", y=title_y, verticalalignment='top', fontsize=20)
    # 中值滤波
    depth = cv2.medianBlur(sample.depth, 5)
    axes[2].imshow(depth, cmap='gray')
    axes[2].set_title("(c) 中值滤波", y=title_y, verticalalignment='top', fontsize=20)
    # 高斯滤波+中值滤波
    depth = cv2.GaussianBlur(sample.depth, ksize=(5, 5), sigmaX=1)
    depth = cv2.medianBlur(depth, 5)
    axes[3].imshow(depth, cmap='gray')
    axes[3].set_title("(d) 高斯滤波+中值滤波", y=title_y, verticalalignment='top', fontsize=20)
    # 中值滤波+高斯滤波
    depth = cv2.medianBlur(sample.depth, 5)
    depth = cv2.GaussianBlur(depth, ksize=(5, 5), sigmaX=1)
    axes[4].imshow(depth, cmap='gray')
    axes[4].set_title("(e) 中值滤波+高斯滤波", y=title_y, verticalalignment='top', fontsize=20)
    
    plt.tight_layout()
    plt.show()
def test_senz_bilateral_filter():
    """
    测试Senz数据集的双边滤波效果
    """
    from main import create_dataset
    dataset: SenzDataset = create_dataset(Config(dataset_type='Senz'))
    
    sample_idx = np.random.randint(len(dataset))
    sample = dataset[sample_idx]

    origin_depth = sample.depth.cpu().numpy() if dataset.to_tensor else sample.depth
    depth = cv2.bilateralFilter(origin_depth, 9, 0, 8)

    # sigma_space, sigma_color = torch.meshgrid(torch.arange(0, 200, 20), torch.arange(0, 8, 1))
    # col_num, row_num = sigma_space.shape
    # sigma_space = sigma_space.numpy()
    # sigma_color = sigma_color.numpy()
    sigma = np.arange(0, 8, 0.5)
    pic_count = len(sigma) + 1
    row_num = math.ceil(math.sqrt(pic_count // 2))
    col_num = math.ceil(pic_count / row_num)

    fig, axes = plt.subplots(row_num, col_num, figsize=(20, 10))
    fig.suptitle(f"Senz数据集样本 {sample_idx} 双边滤波")
    axes[0, 0].imshow(depth, cmap='gray')
    axes[0, 0].set_title("原始深度图")
    depth_list = [None] * pic_count

    for i in range(row_num):
        for j in range(col_num):
            if i == 0 and j == 0:
                continue
            index = i * col_num + j - 1
            if index + 1 >= pic_count:
                break
            axe = axes[i][j]
            # depth = cv2.bilateralFilter(origin_depth, 9, sigma_space[j][i], sigma_color[j][i])
            # depth_list[index] = cv2.bilateralFilter(origin_depth, 9, sigma[index], 0)
            depth_list[index] = cv2.bilateralFilter(origin_depth, 9, 0, sigma[index])
            axe.imshow(depth_list[index], cmap='gray')
            # axe.set_title(f"sigma_space: {sigma_space[j][i]}\nsigma_color: {sigma_color[j][i]}")
            # axe.set_title(f"sigma_color: {sigma[index]}")
            axe.set_title(f"sigma_space: {sigma[index]}")

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
    pred = segmentor.predict_mask_impl(sample.depth, tree)
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
    pred = segmentor.predict_mask_impl(sample.depth, forest, return_prob=return_prob)
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
            predict_mask = segmentor.predict_mask_impl(depth_map, trees[i])
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
    
def test_direct_method(dataset: SenzDataset):
    sample_idx = np.random.randint(len(dataset))
    sample = dataset[sample_idx]

    import torch.hub
    model = torch.hub.load(
        # source='local',
        repo_or_dir='guglielmocamporese/hands-segmentation-pytorch', 
        model='hand_segmentor', 
        pretrained=True,
    )

    # Inference
    model.eval()
    rgb = torch.tensor(sample.color_rgb)
    img_rnd = rgb.permute(2, 1, 0)[None, :, :, :].float() # [B, C, H, W]
    preds: torch.Tensor = model(img_rnd).argmax(1) # [B, H, W]
    preds = preds[0].cpu().numpy().T.astype(np.uint8)

    # 绘制结果
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    axes[1].imshow(preds, cmap='gray')
    axes[1].set_title("Preds")
    plt.show()

def test_segment_all_result(dataset: RHDDataset | SenzDataset, force_predict: bool = False, show_overall: bool = False, show_good: bool = False, show_bad: bool = False):
    """
    测试所有样本的分割效果
    """
    from main import create_segmentor
    segmentor = create_segmentor(Config(seg_type='RDF'))

    file = open('SegModel/RDF/result.csv', 'r+', encoding='utf-8')
    iou_dict = {}
    for line in file.readlines():
        line = line.strip().split(',')
        iou_dict[int(line[0])] = float(line[1])

    sample_count = len(dataset)
    # sample_count = 1000
    for i in tqdm(range(sample_count), desc="测试样本"):
        sample = dataset[i]
        if i not in iou_dict or force_predict:
            pred, center = segmentor.predict_mask_impl(sample.color, sample.depth, return_prob=False)
            iou = SegmentationEvaluation.mean_iou_static(pred, sample.mask)
            iou_dict[i] = iou
            file.write(f"{i},{iou}\n")
    file.close()

    sorted_iou_dict = sorted(iou_dict.items(), key=lambda item: item[1])
    if show_overall:
        plt.plot(np.arange(len(sorted_iou_dict)), [x[1] for x in sorted_iou_dict])
        plt.ylim(0, 1.0)
        plt.xlabel("样本编号")
        plt.ylabel("平均 IoU")
        plt.show()

    if show_good:
        sample_idxs, pred_list = [], []
        search_idx = len(sorted_iou_dict) - 1
        while len(sample_idxs) < 10:
            sample_idx = sorted_iou_dict[search_idx][0]
            if dataset.is_single_hand(sample_idx):
                sample_idxs.append(sample_idx)
            search_idx -= 1
        for idx in sample_idxs:
            sample = dataset[idx]
            pred, center = segmentor.predict_mask_impl(sample.color, sample.depth, return_prob=False)
            pred_list.append(pred)
        __show_multi_segmentation_result(dataset, sample_idxs, pred_list)
    
    if show_bad:
        sample_idxs, pred_list = [], []
        search_idx = 0
        while len(sample_idxs) < 10:
            sample_idx = sorted_iou_dict[search_idx][0]
            if dataset.is_single_hand(sample_idx):
                sample_idxs.append(sample_idx)
            search_idx += 1
        for idx in sample_idxs:
            sample = dataset[idx]
            pred, center = segmentor.predict_mask_impl(sample.color, sample.depth, return_prob=False)
            pred_list.append(pred)
        __show_multi_segmentation_result(dataset, sample_idxs, pred_list, trans=True)

def test_contour_extract(dataset: RHDDataset | SenzDataset, tracker: ThreeDSCTracker):
    """
    测试轮廓提取
    """
    sample_idx = np.random.randint(len(dataset))
    # sample_idx = 0
    sample = dataset[sample_idx]

    pred = torch.tensor(sample.mask)
    contour_idx = tracker.extract_contour(sample.depth, pred)
    contour = np.zeros_like(sample.depth.cpu())
    contour[contour_idx[:, 0], contour_idx[:, 1]] = 1
    # color = transforms.Grayscale(num_output_channels=1)(sample.color.permute(2, 0, 1))[0]
    # color_contour = tracker.extract_contour(color, pred)
    # depth_contour = tracker.extract_contour(pred, pred)
    # plt.hist(depth_contour.cpu().numpy().flatten(), bins=100, range=(0, 1), alpha=0.75)
    # plt.show()
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 10))

    axes[0].imshow(sample.depth.cpu(), cmap='gray')
    axes[0].set_title("原始图像")
    axes[1].imshow(sample.mask, cmap='gray')
    axes[1].set_title("Ground Truth")

    axes[2].imshow(contour.cpu().numpy(), cmap='gray')
    axes[2].set_title("深度图轮廓提取")
    # axes[3].imshow(color_contour.cpu().numpy(), cmap='gray')
    # axes[3].set_title("彩色图轮廓提取")

    plt.tight_layout()
    plt.show()
def test_simplify_contour(dataset: RHDDataset | SenzDataset, tracker: ThreeDSCTracker):
    """
    测试简化轮廓算法效果
    """
    sample_idx = np.random.randint(len(dataset))
    # sample_idx = 94
    sample = dataset[sample_idx]

    pred = torch.tensor(sample.mask)
    contour = tracker.extract_contour(sample.depth, pred)
    cv2_sim_contour = tracker.extract_contour(sample.depth, pred, cv2_method=cv2.CHAIN_APPROX_SIMPLE)
    
    target_points = [300, 150, 80, 50, 30]
    row_count = 1
    col_count = 3
    col_count = len(target_points) + 1
    plt.figure(figsize=(22, 12))
    plt.suptitle(f"简化轮廓算法效果 样本 {sample_idx}")

    # mask_axe = plt.subplot(row_count, col_count, 1)
    # mask_axe.imshow(sample.mask, cmap='gray')
    # mask_axe.set_title("手部分割掩膜")

    origin_axe = plt.subplot(row_count, col_count, 1)
    __set_contour_axe(origin_axe, contour, f"原始轮廓\n点数量：{contour.shape[0]}")
    # cv2_axe = plt.subplot(row_count, col_count, 3)
    # __set_contour_axe(cv2_axe, cv2_sim_contour, f"cv2 简化轮廓\n点数量：{cv2_sim_contour.shape[0]}")

    for i in range(len(target_points)):
        tracker.contour_max_points = target_points[i]
        sim_contour = tracker.simplify_contour(contour)
        new_axe = plt.subplot(row_count, col_count, i + (row_count - 1) * col_count + 2)
        __set_contour_axe(new_axe, sim_contour, f"点数量：{sim_contour.shape[0]}")

    plt.tight_layout()
    plt.show()
def test_descriptor_features(dataset: RHDDataset | SenzDataset, tracker: ThreeDSCTracker):
    """
    测试获取3D-SC描述符特征
    """
    sample_idx = np.random.randint(len(dataset))
    # sample_idx = 0
    sample = dataset[sample_idx]

    pred = torch.tensor(sample.mask)
    depth_contour = tracker.extract_contour(sample.depth, pred)
    simplified_contour = tracker.simplify_contour(depth_contour)
    features = tracker.get_descriptor_features(simplified_contour, sample.depth)

    # 绘制特征图
    plt.figure(figsize=(10, 5))
    plt.suptitle(f"3D-SC描述符特征 样本 {sample_idx}")

    depth_axe = plt.subplot(1, 2, 1)
    depth_axe.set_title("原始图像")
    depth_axe.imshow(sample.depth.cpu(), cmap='gray')

    feature_axe = plt.subplot(1, 2, 2)
    __set_feature_axe(feature_axe, features)

    plt.tight_layout()
    plt.show()
def test_feature_direction(dataset: RHDDataset | SenzDataset):
    """
    测试特征方向
    """
    from main import create_tracker
    tracker = create_tracker(Config(track_type='3DSC'))

    sample_idx = np.random.randint(len(dataset))
    # sample_idx = 359
    sample = dataset[sample_idx]

    def set_direction(axe: plt.Axes, start: np.ndarray, offset: np.ndarray, mask: np.ndarray | None = None, arrow_mask: np.ndarray | None = None):
        """
        显示特征方向
        :param axes: 图像对象
        :param start: 箭头起点, (point_count, 2)
        :param offset: 箭头向量, (point_count, 2)
        :param mask: 掩膜图像, (H, W)
        :param arrow_mask: 选择显示箭头
        """
        mask = mask if mask is not None else sample.mask
        X, Y = start[:, 0], start[:, 1]
        offset = offset / np.linalg.norm(offset, axis=1, keepdims=True) * (mask.shape[0] // 18)
        U, V = offset[:, 0], offset[:, 1]
        axe.plot(np.concatenate([X, X[0:1]]), np.concatenate([Y, Y[0:1]]), color='black', marker=',')
        if arrow_mask is not None:
            X, Y, U, V = X[arrow_mask], Y[arrow_mask], U[arrow_mask], V[arrow_mask]
        axe.quiver(X, Y, U, V, 
                    angles='xy', scale_units='xy', scale=1, 
                    color='red', 
                    units='width', width=0.005, headlength=2, headwidth=2)
        axe.set_axis_off()
        axe.set_aspect('equal')
        # axe.set_xlim(0, mask.shape[1])
        # axe.set_ylim(mask.shape[0], 0)
        # axe.imshow(mask, cmap='gray')

    def set_direction_comp(axes: List[plt.Axes], contour: np.ndarray, mask: np.ndarray, show_title: bool = False, arrow_mask: np.ndarray | None = None):
        """
        显示不同方法获取的特征方向对比
        :param axes: 图像对象列表
        :param contour: 轮廓点, (point_count, 2)
        :param mask: 掩膜图像, (H, W)
        """
        idx = 0
        for_c = np.concatenate([contour, contour[0:1]], axis=0)
        for_offset = for_c[1:] - for_c[:-1]
        # set_direction(axes[idx], contour, for_offset, mask, arrow_mask)
        # if show_title:
        #     axes[idx].set_title("前向差分")
        # idx += 1

        back_c = np.concatenate([contour[-2:-1], contour], axis=0)
        back_offset = back_c[:-1] - back_c[1:]
        # set_direction(axes[idx], contour, back_offset, mask, arrow_mask)
        # if show_title:
        #     axes[idx].set_title("后向差分")
        # idx += 1

        mid_offset = for_c[1:] - back_c[:-1]
        for_offset = for_offset / np.linalg.norm(for_offset, axis=1, keepdims=True)
        back_offset = back_offset / np.linalg.norm(back_offset, axis=1, keepdims=True)
        # mid_offset = (for_offset + back_offset) / 2
        # mid_offset = np.stack([-mid_offset[:, 1], mid_offset[:, 0]], axis=1)
        # mid_offset = (for_offset - back_offset) / 2
        set_direction(axes[idx], contour, mid_offset, mask, arrow_mask)
        if show_title:
            axes[idx].set_title("中值差分")
        idx += 1

        center = contour.mean(axis=0)
        center_offset = contour - center
        set_direction(axes[idx], contour, center_offset, mask, arrow_mask)
        if show_title:
            axes[idx].set_title("质心参考点")
        idx += 1

    temp = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey=True)
    axes: List[plt.Axes] = temp[1]
    temp[0].suptitle(f"特征方向测试 样本 {sample_idx}")
    
    arrow_mask = np.random.choice(tracker.contour_max_points, 10, replace=False)
    arrow_mask = None
    for i in range(4):
        rotation = lambda x: transforms.RandomRotation(degrees=(90 * i,90 * i))(x.unsqueeze(0)).squeeze(0)
        pred: torch.Tensor = rotation(torch.tensor(sample.mask))
        depth: torch.Tensor  = rotation(sample.depth)
        contour = tracker.extract_contour(depth, pred)
        sim_contour = tracker.simplify_contour(contour)
        # 噪音干扰
        sim_contour = sim_contour.astype(np.float64) + np.random.normal(1, 1, size=sim_contour.shape)
        sim_contour = sim_contour.astype(np.int32)
        set_direction_comp(axes[:, i], sim_contour, pred.cpu().numpy(), False, arrow_mask)

    plt.show()

def test_lbp_features(dataset: RHDDataset | SenzDataset, segmentor: RDFSegmentor):
    """
    测试LBP特征
    """
    # sample_idx = np.random.randint(len(dataset))
    sample_idx = 0
    sample = dataset[sample_idx]

    from main import create_tracker
    tracker = create_tracker(Config(track_type='LBP'))

    pred = torch.tensor(sample.mask if isinstance(dataset, RHDDataset) else segmentor.predict_mask(sample.color, sample.depth))
    # pred = torch.tensor(sample.mask)
    lbp = tracker.get_lbp_map(sample.depth, pred)
    # lbp = tracker(sample.depth, torch.ones_like(sample.depth, device=sample.depth.device))
    hist = tracker.get_lbp_hist(lbp, pred)

    # 绘制特征图
    plt.figure(figsize=(10, 5))
    plt.suptitle(f"LBP特征 样本 {sample_idx}")

    depth_axe = plt.subplot(1, 3, 1)
    depth_axe.set_title("原始深度图像")
    depth_axe.imshow(sample.depth.cpu(), cmap='gray')

    feature_axe = plt.subplot(1, 3, 2)
    feature_axe.set_title("LBP特征")
    feature_axe.imshow(lbp.cpu().numpy(), cmap='gray')

    hist_axe = plt.subplot(1, 3, 3)
    hist_axe.set_title("LBP直方图")
    hist_axe.bar(np.arange(len(hist)), hist.cpu().numpy(), width=1.0)

    plt.tight_layout()
    plt.show()

def test_feature_effect(feature_name: str, output_dir: str = './SegModel'):
    from main import create_dataset
    dataset = create_dataset(Config(dataset_type='Senz'))

    file_name = f'{type(dataset).__name__}_{dataset.set_type}_{feature_name}'
    distance_path = os.path.join(output_dir, f'distance_{file_name}.npy')

    labels = dataset.labels
    distance = np.load(distance_path)

    mean = DTWClassifier.get_mean_distance(distance, labels)
    rela_dist = mean[:, 0] / mean[:, 1]
    print(f"平均距离: {np.mean(rela_dist):.4f}")
def test_feature_invariance(tracker: ThreeDSCTracker | LBPTracker):
    """
    测试特征不变性
    """
    from main import create_dataset
    dataset = create_dataset(Config(dataset_type='RHD'))
    # sample_idx = np.random.randint(len(dataset))

    def show_feature_3dsc(sample_idx: int, axes: List[plt.Axes], trans: Callable[[torch.Tensor], torch.Tensor] | None = None, point_idx: int = 0, dont_trans_mask: bool = False):
        sample = dataset[sample_idx]
        sample.mask = torch.tensor(sample.mask)
        if trans is None:
            trans = lambda x: x
        depth_map = trans(sample.depth.unsqueeze(0)).squeeze(0)
        mask = sample.mask if dont_trans_mask else trans(sample.mask.unsqueeze(0)).squeeze(0)

        contour = tracker.extract_contour(depth_map, mask)
        sim_contour = tracker.simplify_contour(contour)
        features = tracker.get_descriptor_features(sim_contour, depth_map)

        point = sim_contour[point_idx]
        feature = features[point_idx].cpu().numpy()

        # 显示深度图
        axes[0].imshow(depth_map.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[0].set_xlim(0, sample.depth.shape[1])
        axes[0].set_ylim(sample.depth.shape[0], 0)
        axes[0].set_axis_off()
        # 显示mask
        axes[1].imshow(mask.cpu().numpy(), cmap='gray')
        axes[1].scatter([point[0]], [point[1]], c='r', linewidths=3)
        axes[1].set_xlim(0, sample.mask.shape[1])
        axes[1].set_ylim(sample.mask.shape[0], 0)
        axes[1].set_axis_off()
        # axes[1].set_axis_off()
        # axes[1].scatter(sim_contour[:, 0].cpu(), sim_contour[:, 1].cpu(), c='r', linewidths=3)
        # axes[1].scatter(contour[:, 0].cpu(), contour[:, 1].cpu(), c='r', linewidths=3)
        # for i in range(len(sim_contour)):
        #     axes[1].text(sim_contour[i][0], sim_contour[i][1], str(i), color='r', fontsize=10)
        # 显示特征
        __set_feature_axe(axes[2], feature, show_title=False)

    show_func = show_feature_3dsc if isinstance(tracker, ThreeDSCTracker) else None

    row, col = 5, 3
    axes = [0, 1, 2, 3, 4]
    sample_idx = 359
    fig, axes = plt.subplots(row, col, figsize=(12, 12))
    # fig.suptitle(f"特征不变性测试 样本 {sample_idx}")

    show_func(sample_idx, axes[0], None, 31)
    show_func(sample_idx, axes[1], transforms.RandomRotation(degrees=(90,90)), 71)
    show_func(sample_idx, axes[2], transforms.Resize((160, 160), antialias=False), 27)
    show_func(sample_idx, axes[3], lambda x: torch.clamp_max(x + 0.5, 1.0), 31, dont_trans_mask=True)
    show_func(sample_idx, axes[4], transforms.Compose([
        transforms.Resize((160, 160), antialias=False),
        transforms.RandomRotation(degrees=(90,90)),
        lambda x: torch.clamp_max(x + 0.5, 1.0)
    ]), 67)

    # plt.tight_layout()
    axes[0, 0].set_title("深度图像")
    axes[0, 1].set_title("手部掩膜与轮廓点")
    axes[0, 2].set_title("3D-SC特征")
    plt.subplots_adjust(wspace=0.05)
    plt.show()

def test_dtw_distance(dataset: RHDDataset | SenzDataset, tracker: ThreeDSCTracker, classifier: DTWClassifier):
    """
    测试 DTW 距离计算结果
    """
    sample_idx = [np.random.randint(len(dataset)) for _ in range(2)]
    # sample_idx = 0
    samples = [dataset[idx] for idx in sample_idx]

    features_list: List[torch.Tensor] = []
    for i in range(len(sample_idx)):
        sample = samples[i]
        pred = torch.tensor(sample.mask)
        depth_contour = tracker.extract_contour(sample.depth, pred)
        simplified_contour = tracker.simplify_contour(depth_contour)
        features = tracker.get_descriptor_features(simplified_contour, sample.depth)
        features_list.append(features)
        
    flatten_features = [features.cpu().numpy().reshape(features.shape[0], -1) for features in features_list]
    distance, path = classifier.get_distance(flatten_features[0], flatten_features[1])
    
    # 将path 展示为折线图
    # plt.figure(figsize=(10, 5))
    # plt.suptitle(f"DTW 距离计算结果 样本 {sample_idx}")

    # depth_axe = plt.subplot(1, 2, 1)
    # depth_axe.set_title("原始图像")
    # depth_axe.imshow(samples[0].depth.cpu(), cmap='gray')

    # feature_axe = plt.subplot(1, 2, 2)
    # _set_feature_axe(feature_axe, features_list[0])

    plt.plot(path[0], path[1])
    plt.title(f"样本 {sample_idx[0]} 到样本 {sample_idx[1]} 的DTW路径\n距离: {distance:.4f}")

    plt.show()
def test_custom_fastdtw(dataset: RHDDataset | SenzDataset, feature_dir: str):
    """
    测试自定义的DTW算法效果
    """
    sample_idx = [np.random.randint(len(dataset)) for _ in range(2)]
    sample_idx = [0, 1]
    # samples = [dataset[idx] for idx in sample_idx]

    features_list: List[torch.Tensor] = []
    file_path = __get_feature_path(feature_dir, type(dataset).__name__, dataset.set_type)
    for idx in sample_idx:
        feature, _, _ = DTWClassifier.from_file(file_path, idx)
        features_list.append(feature.repeat(10, 0))
        # features_list.append(feature)

    fastdtw_cls = DTWClassifier(dtw_type='fastdtw')
    custom_cls = DTWClassifier(dtw_type='custom')

    timer = time.perf_counter()
    fastdtw_dist, _ = fastdtw_cls.get_distance(features_list[0], features_list[1])
    fastdtw_time = time.perf_counter() - timer

    # timer = time.perf_counter()
    # gpu_dist, _ = custom_cls.get_distance_gpu(torch.tensor(features_list[0]).to(custom_cls.device), torch.tensor(features_list[1]).to(custom_cls.device))
    # gpu_time = time.perf_counter() - timer

    timer = time.perf_counter()
    custom_dist, _ = custom_cls.get_distance(features_list[0], features_list[1])
    custom_time = time.perf_counter() - timer

    # print(f"fastdtw 耗时: {fastdtw_time:.4f}s 自定义DTW算法耗时: {custom_time:.4f}s (numpy) {gpu_time:.4f}s (gpu)")
    print(f"fastdtw 耗时: {fastdtw_time:.4f}s 自定义DTW算法耗时: {custom_time:.4f}s")
    min_timer_str = str.format("{:.4f}s", FastDTW.min_timer) if FastDTW.min_timer != 0 else '未计时'
    dist_timer_str = str.format("{:.4f}s", FastDTW.dist_timer) if FastDTW.dist_timer != 0 else '未计时'
    extend_timer_str = str.format("{:.4f}s", FastDTW.extend_timer) if FastDTW.extend_timer != 0 else '未计时'
    print(f"取最小值耗时: {min_timer_str} 计算距离耗时: {dist_timer_str} 扩展索引耗时: {extend_timer_str}")
    assert abs(fastdtw_dist - custom_dist) < (fastdtw_dist + custom_dist) * 0.01, f"自定义DTW算法与fastdtw算法结果不一致 fastdtw: {fastdtw_dist:.4f} 自定义: {custom_dist:.4f}"

def test_feature_load(dataset: RHDDataset | SenzDataset, tracker: ThreeDSCTracker, classifier: DTWClassifier, file_dir: str):
    """
    测试加载特征文件
    """
    sample_idx = np.random.randint(10)
    # sample_idx = 0
    sample = dataset[sample_idx]

    calc_features = tracker(sample.depth, torch.tensor(sample.mask))
    calc_features: np.ndarray = calc_features.cpu().numpy().reshape(calc_features.shape[0], -1)
    
    file_path = __get_feature_path(file_dir, type(dataset).__name__, dataset.set_type)
    feature, _, _ = DTWClassifier.from_file(file_path, sample_idx)
    assert np.abs(calc_features - feature).max() < 1e-6, "特征计算结果与加载结果不一致"

    features, _, _ = DTWClassifier.from_file_all(file_path)
    assert np.abs(features[sample_idx] - calc_features).max() < 1e-6, "特征计算结果与加载结果不一致"
def test_fastdtw_speed(dataset: RHDDataset | SenzDataset, feature_dir: str):
    """
    测试fastdtw算法速度
    """
    file_name = f'{type(dataset).__name__}_{dataset.set_type}_ThreeDSCTracker'
    feature_file = f"features_{file_name}.bin"
    features, _, _ = DTWClassifier.from_file_all(os.path.join(feature_dir, feature_file))

    classifier = DTWClassifier()
    def test_speed(dtw_type: str):
        classifier.dtw_type = dtw_type
        timer = time.perf_counter()
        for i in tqdm(range(10000), desc=f"测试{dtw_type}算法速度"):
            idx = np.random.randint(len(features), size=2)
            classifier.get_distance(features[idx[0]], features[idx[1]])
        timer = time.perf_counter() - timer
        print(f"{dtw_type}算法耗时: {timer:.4f}s")

    test_speed('fastdtw')
    test_speed('custom')
    test_speed('tslearn')
    test_speed('tslearn-fake')

    # fig = plt.subplot(1, 1, 1)
    # bar = plt.bar(range(5), [596.06, 187.41, 80.85, 203.56, 5.26], color=['#23aaf2', '#23aaf2', '#138a07', '#138a07', '#2f3a4c'])
    # fig.spines['top'].set_visible(False)
    # fig.spines['right'].set_visible(False)
    # plt.bar_label(bar, labels=[596.06, 187.41, 80.85, 203.56, 5.26], padding=3)
    # plt.xticks(range(5), ['tslearn', 'fastdtw', '本文\nnumpy', '本文\npytorch', 'tslearn\n欧氏距离'])
    # plt.show()

def test_classify_result():
    """
    测试分类结果
    """
    from main import create_dataset, create_classifier
    dataset = create_dataset(Config(dataset_type='Senz'))
    classifier = create_classifier(Config(cls_type='DTW'))

    # 统计数据
    n_cls = 11
    n_round = 10
    test_ratio = 0.3

    data = np.zeros((n_cls, n_cls)) # data[实际标签, 预测标签]
    for _ in range(n_round):
        idx = np.arange(len(dataset))
        np.random.shuffle(idx)
        test_idx = idx[:int(len(dataset) * test_ratio)]
        labels = [dataset.labels[i] for i in test_idx]
        for label in labels:
            pred_label = classifier(label)
            data[label - 1, pred_label - 1] += 1

    accuracy = np.trace(data) / data.sum()
    data = data / data.sum(axis=1, keepdims=True) * 100
    data_str = [[(f'{data[i, j]:.2f}' if data[i, j]!= 0 else '')
                    for j in range(n_cls)] 
                    for i in range(n_cls)]
    
    # 绘制表格
    temp = plt.subplots()
    plt.suptitle(f"分类结果 准确率: {accuracy * 100:.4f}%")
    ax: plt.Axes = temp[1]
    colLabels = [f'P{i + 1}' for i in range(n_cls)]
    rowLabels = [f'G{i + 1}' for i in range(n_cls)]
    table = ax.table(cellText=data_str, 
                     colLabels=colLabels, rowLabels=rowLabels, 
                     loc='center')

    # 调整表格比例
    table.scale(1, 1)
    # 调整颜色
    for cls in range(n_cls):
        cell = table[cls + 1, cls]
        cell.set_color('#138a07')
        cell.set_fill(True)
        cell.set_text_props(color='white')

    # 隐藏列标题和行标题的边框
    celld = table.get_celld()
    for cell in celld:
        if cell[0] == 0 or cell[1] == -1:
            celld[cell].set_edgecolor('none')
    # 隐藏原有的坐标轴
    ax.axis('off')

    # 显示图表
    plt.show()


def __show_segmentation_result(sample: RHDDatasetItem | SenzDatasetItem, pred: np.ndarray, binary: bool = True, color: bool = False):
    """
    显示分割结果
    :param sample: 样本数据
    :param pred: 预测结果 (width, height)
    """
    mask_is_conf = hasattr(sample, 'confidence')
    if mask_is_conf:
        mask_color = sample.confidence
    else:
        mask_color = np.zeros((sample.mask.shape[0], sample.mask.shape[1], 3), dtype=np.uint8)
        mask_color[sample.mask == 0] = [0, 0, 0]  # 背景
        mask_color[sample.mask == 1] = [255, 255, 255]  # 手部

    if binary:
        pred_color = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        pred_color[pred == 0] = [0, 0, 0]  # 背景
        pred_color[pred == 1] = [255, 255, 255]  # 手部
    else:
        pred_color = pred / 255.0

    temp = plt.subplots(1, 3, figsize=(10, 5))
    iou = SegmentationEvaluation.mean_iou_static(pred, sample.mask)
    temp[0].suptitle(f"分割结果 IoU: {iou:.4f}")
    axes: List[plt.Axes] = temp[1]
    if color:
        axes[0].imshow(sample.color_rgb)
    else:
        axes[0].imshow(sample.depth.cpu(), cmap='gray')
    axes[0].set_title("原始图像")
    axes[1].imshow(mask_color, cmap='gray' if mask_is_conf else None)
    axes[1].set_title("Ground Truth")
    axes[2].imshow(pred_color, cmap='gray' if not binary else None)
    axes[2].set_title("预测结果")
    for ax in axes:
        ax.set_axis_off()

    plt.tight_layout()
    plt.show()
def __show_multi_segmentation_result(dataset: RHDDataset, sample_idx: List[int], pred: List[np.ndarray], trans: bool = False):
    """
    显示多个分割结果
    """
    assert len(sample_idx) == len(pred), "预测结果数量不正确"
    n_row, n_col = 4, len(sample_idx)
    if trans:
        n_row, n_col = n_col, n_row

    fig, axes = plt.subplots(n_row, n_col, figsize=(22, 12))
    for i, idx in enumerate(sample_idx):
        sample = dataset[idx]
        sample_axes: List[plt.Axes] = axes[i] if trans else axes[:, i]
        sample_axes[0].imshow(sample.color_rgb)
        if not trans:
            sample_axes[0].set_title(f"样本 {idx}")
        sample_axes[1].imshow(sample.depth.cpu(), cmap='gray')
        sample_axes[2].imshow(sample.mask, cmap='gray')
        sample_axes[3].imshow(pred[i], cmap='gray')
        for ax in sample_axes:
            ax.set_axis_off()
    
    plt.tight_layout()
    plt.show()


def __set_contour_axe(axe: plt.Axes, contour: np.ndarray, title: str | None = "轮廓图", shape: Tuple[int, int] | None = None, fontsize: int = 20):
    """
    显示轮廓图
    """
    if title is not None:
        axe.set_title(title, fontsize=fontsize)
    axe.plot(list(contour[:, 0]) + [contour[0, 0]], list(contour[:, 1]) + [contour[0, 1]], 
             marker='o', markersize=4, 
             linewidth=1, color='black', 
             scalex=False, scaley=False)
    if shape is not None:
        axe.set_xlim(0, shape[1])
        axe.set_ylim(shape[0], 0)
    else:
        axe.set_xlim(contour[:, 0].min() - 10, contour[:, 0].max() + 10)
        axe.set_ylim(contour[:, 1].max() + 10, contour[:, 1].min() - 10)
    axe.set_aspect('equal')
    axe.axis('off')

def __set_feature_axe(axe: plt.Axes, features: torch.Tensor | np.ndarray, show_title: bool=True):
    """
    显示特征图
    """
    image = __get_feature_image(features if len(features.shape) == 2 else features[0])
    if show_title:
        axe.set_title("特征图")
    axe.imshow(image, cmap='gray')
    axe.set_xticks(np.arange(0, features.shape[-1], 1) * 100 + 50)
    axe.set_xticklabels(np.arange(0, features.shape[-1], 1))
    axe.set_xlabel("角度区间")
    axe.set_yticks(np.arange(0, features.shape[-2], 1) * 100 + 50)
    axe.set_yticklabels(np.arange(0, features.shape[-2], 1))
    axe.set_ylabel("长度区间")

def __get_feature_image(features: torch.Tensor | np.ndarray) -> np.ndarray:
    """
    获取特征显示图片
    """
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    features = features / max(features.max(), 1)
    return np.repeat(features, 100, axis=0).repeat(100, axis=1)

def __get_feature_path(file_dir: str, dataset_type: str, dataset_split: str) -> str:
    return f"{file_dir}/features_{dataset_type}_{dataset_split}.bin"