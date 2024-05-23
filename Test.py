# 这个文件中主要是用来进行一些实验的
import os
import time
import math
import pickle
from typing import List, Tuple
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
from Segmentation import RDFSegmentor, DecisionTree, ResNetSegmentor
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


def test_senz_dataset(dataset: SenzDataset):
    """
    测试Senz数据集导入效果
    """ 
    sample_idx = np.random.randint(len(dataset))
    sample = dataset[sample_idx]

    fig, axes = plt.subplots(3, 4, figsize=(20, 8))
    fig.suptitle(f"Senz数据集样本 {sample_idx}")

    axes[0, 0].imshow(sample.color_rgb)
    axes[0, 0].set_title("RGB")
    axes[1, 0].imshow(sample.depth.cpu(), cmap='gray')
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

def test_senz_bilateral_filter(dataset: SenzDataset):
    """
    测试Senz数据集的双边滤波效果
    """
    sample_idx = np.random.randint(len(dataset))
    sample = dataset[sample_idx]

    origin_depth = sample.depth.cpu().numpy() if dataset.to_tensor else sample.depth

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
    axes[0, 0].imshow(origin_depth, cmap='gray')
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
def test_resnet_predict(dataset: SenzDataset | RHDDataset, segmentor: ResNetSegmentor, return_prob: bool = False):
    """
    测试 ResNet 的预测效果
    """
    # sample_idx = np.random.randint(len(dataset))
    sample_idx = 0
    sample = dataset[sample_idx]

    # 预测结果
    pred, center = segmentor.predict_mask_impl(sample.color, sample.depth, return_prob)
    __show_segmentation_result(sample, pred, not return_prob)

def test_predict_roi(dataset: SenzDataset | RHDDataset, segmentor: RDFSegmentor | ResNetSegmentor):
    """
    测试预测手部ROI区域
    """
    sample_idx = np.random.randint(len(dataset))
    sample = dataset[sample_idx]

    # 预测结果
    bbox = segmentor.get_roi_bbox(sample.color)
    bbox = bbox if bbox is not None else (0, 0, 1, 1)
    bbox = (int(bbox[0] * sample.color.shape[1]), int(bbox[1] * sample.color.shape[0]),
            int(bbox[2] * sample.color.shape[1]), int(bbox[3] * sample.color.shape[0]))
    img = sample.color.cpu().numpy()

    plt.suptitle(f"预测手部ROI区域 样本 {sample_idx}")
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("原图")

    plt.subplot(1, 2, 2)
    plt.imshow(img[bbox[1]:bbox[3], bbox[0]:bbox[2]])
    plt.title(f"ROI区域\n{bbox}")
    # x_min, y_min, x_max, y_max = bbox
    # width = x_max - x_min
    # height = y_max - y_min
    # rect = Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='r', facecolor='none')
    # plt.gca().add_patch(rect)

    plt.show()
def test_segement_with_roi(dataset: SenzDataset | RHDDataset, segmentor: RDFSegmentor | ResNetSegmentor, return_prob: bool = False):
    """
    测试预测手部ROI区域后的分割效果
    """
    sample_idx = 0
    sample = dataset[sample_idx]

    # 直接预测结果
    segmentor.roi_detection = False
    pred_direct = segmentor.predict_mask(sample.color, sample.depth, return_prob)

    # 使用ROI预测结果
    segmentor.roi_detection = True
    roi = ResNetSegmentor.bbox_to_indices(segmentor.get_roi_bbox(sample.color), sample.color.shape[:2])
    pred_roi = segmentor.predict_mask(sample.color, sample.depth, return_prob)

    # 显示预测结果
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle(f"预测手部ROI区域后的分割效果 样本 {sample_idx}")

    axes[0, 0].imshow(sample.color.cpu().numpy())
    axes[0, 0].set_title("原图")
    axes[0, 1].imshow(pred_direct, cmap='gray')
    axes[0, 1].set_title("直接预测结果")

    axes[1, 0].imshow(sample.color.cpu().numpy()[roi[1]:roi[3], roi[0]:roi[2]])
    axes[1, 0].set_title("ROI区域")
    axes[1, 1].imshow(pred_roi, cmap='gray')
    axes[1, 1].set_title("ROI预测结果")

    plt.show()


def test_contour_extract(dataset: RHDDataset | SenzDataset, tracker: ThreeDSCTracker):
    """
    测试轮廓提取
    """
    sample_idx = np.random.randint(len(dataset))
    # sample_idx = 0
    sample = dataset[sample_idx]

    pred = torch.tensor(sample.mask)
    depth_contour = tracker.extract_contour(sample.depth, pred)
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

    axes[2].imshow(depth_contour.cpu().numpy(), cmap='gray')
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
    # sample_idx = 0
    sample = dataset[sample_idx]

    pred = torch.tensor(sample.mask)
    depth_contour = tracker.extract_contour(sample.depth, pred)
    simplified_contour = tracker.simplify_contour(depth_contour).cpu().numpy()

    new_contour = np.zeros_like(sample.depth.cpu())
    new_contour[simplified_contour[:, 0], simplified_contour[:, 1]] = 1

    contour_depth = sample.depth.cpu().numpy()[simplified_contour[:, 0], simplified_contour[:, 1]]
    
    figure_count = 3
    plt.figure(figsize=(15, 7))
    plt.title(f"简化轮廓算法效果 样本 {sample_idx}")
    plt.subplot(1, figure_count, 1)
    plt.imshow(sample.depth.cpu(), cmap='gray')
    plt.suptitle("原始深度图像")

    plt.subplot(1, figure_count, 2)
    plt.imshow(depth_contour.cpu(), cmap='gray')
    plt.suptitle("提取的轮廓")

    plt.subplot(1, figure_count, 3)
    plt.imshow(new_contour, cmap='gray')
    plt.suptitle("简化后的轮廓")

    # plt.subplot(1, figure_count, 4)
    # plt.bar(np.arange(len(contour_depth)), contour_depth)
    # plt.suptitle("轮廓的深度")

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
    _set_feature_axe(feature_axe, features)

    plt.tight_layout()
    plt.show()


def test_lbp_features(dataset: RHDDataset | SenzDataset, segmentor: RDFSegmentor | ResNetSegmentor):
    """
    测试LBP特征
    """
    # sample_idx = np.random.randint(len(dataset))
    sample_idx = 0
    sample = dataset[sample_idx]

    from main import create_tracker
    tracker = create_tracker(Config(track_type='LBP'))

    pred = torch.tensor(sample.mask if isinstance(dataset, RHDDataset) else segmentor.predict_mask(sample.color, sample.depth))
    lbp = tracker(sample.depth, pred)
    # lbp = tracker(sample.depth, torch.ones_like(sample.depth, device=sample.depth.device))
    
    # 绘制特征图
    plt.figure(figsize=(10, 5))
    plt.suptitle(f"LBP特征 样本 {sample_idx}")

    depth_axe = plt.subplot(1, 2, 1)
    depth_axe.set_title("原始深度图像")
    depth_axe.imshow(sample.depth.cpu(), cmap='gray')

    feature_axe = plt.subplot(1, 2, 2)
    feature_axe.set_title("LBP特征")
    feature_axe.imshow(lbp.cpu().numpy(), cmap='gray')

    plt.tight_layout()
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
        
    flatten_features = [features.reshape(features.shape[0], -1) for features in features_list]
    distance, path = classifier.get_distance_gpu(flatten_features[0], flatten_features[1])
    
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

def __show_segmentation_result(sample: RHDDatasetItem | SenzDatasetItem, pred: np.ndarray, binary: bool = True):
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
    axes[1].imshow(mask_color, cmap='gray' if mask_is_conf else None)
    axes[1].set_title("Ground Truth")
    axes[2].imshow(pred_color, cmap='gray' if not binary else None)
    axes[2].set_title("预测结果")

    plt.tight_layout()
    plt.show()

def _set_feature_axe(axe: plt.Axes, features: torch.Tensor | np.ndarray):
    """
    显示特征图
    """
    image = __get_feature_image(features if len(features.shape) == 2 else features[0])
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