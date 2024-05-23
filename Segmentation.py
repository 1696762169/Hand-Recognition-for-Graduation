import os
from typing import List, Tuple
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from scipy import stats

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import ResNet.main as ResNet
import YOLO.predict as YOLO

from tqdm import tqdm
from matplotlib import pyplot as plt
import logging

from Dataset.RHD import RHDDataset, RHDDatasetItem
from Config import Config
from Evaluation import SegmentationEvaluation

class CustomSegmentor(ABC):
    """
    自定义分割方法基类
    """
    def __init__(self, 
                 model_path: str, 
                 roi_detection: bool=True, 
                 roi_extended: float = 0.0):
        self.model_path = model_path
        self.roi_detection = roi_detection
        self.roi_extended = roi_extended
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 网格索引缓存
        self.grid_indices: torch.Tensor = None  # (2, w, h)
        self.flatten_indices: torch.Tensor = None  # (2, w*h)
        # ROI检测模型
        self.roi_model = YOLO.load_model(
            model_path="YOLO/weights/hand_416_good.pt",
            cfg="yolo",
            data_cfg="YOLO/cfg/hand.data", 
            img_size=416)

    @abstractmethod
    def predict_mask(self, rgb_map: torch.Tensor, depth_map: torch.Tensor, return_prob: bool=False) -> np.ndarray:
        """
        预测掩码
        :param rgb_map: RGB图 ([batch_size], h, w, 3) [0, 255]
        :param depth_map: 深度图 ([batch_size], h, w) [0, 1]
        :param return_prob: 是否返回预测概率 否则返回众数（分类结果）
        :return: 分类掩膜 ([batch_size], h, w)
        """

    def buffer_grid_indices(self, shape: Tuple[int, int]) -> None:
        """
        计算并缓存网格索引
        :param shape: 图像的尺寸 (w, h)
        """
        if self.grid_indices is not None and self.flatten_indices is not None and self.grid_indices.shape[1:] == shape:
            return
        indices_x = torch.arange(shape[0], device=self.device)
        indices_y = torch.arange(shape[1], device=self.device)
        self.grid_x, self.grid_y = torch.meshgrid(indices_x, indices_y, indexing='ij')
        self.grid_indices = torch.stack([self.grid_x, self.grid_y], dim=0)
        self.flatten_indices = torch.stack([self.grid_x.flatten(), self.grid_y.flatten()], dim=0)

    def get_roi_bbox(self, rgb_map: torch.Tensor) -> Tuple[float, float, float, float] | None:
        """
        获取ROI的边界框
        :param rgb_map: RGB图 (h, w, 3) [0, 255]
        :return: 边界框相对值 (x1, y1, x2, y2) None表示未检测到ROI
        """
        if not self.roi_detection:
            return None
        # 预测ROI
        porb, bbox = YOLO.predict_image(rgb_map, self.roi_model)
        if porb < 0.1:
            return None
        bbox = [bbox[0] / rgb_map.shape[1], bbox[1] / rgb_map.shape[0], bbox[2] / rgb_map.shape[1], bbox[3] / rgb_map.shape[0]]
        
        # 扩展ROI
        if self.roi_extended != 0:
            bbox[0] = max(0, bbox[0] - self.roi_extended)
            bbox[1] = max(0, bbox[1] - self.roi_extended)
            bbox[2] = min(1, bbox[2] + self.roi_extended)
            bbox[3] = min(1, bbox[3] + self.roi_extended)
        return tuple(bbox)
    
    @staticmethod
    def bbox_to_indices(bbox: Tuple[float, float, float, float], shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        边界框相对值转为索引
        :param bbox: 边界框相对值 (x1, y1, x2, y2)
        :param shape: 图像的尺寸 (h, w)
        :return: 边界框索引 (x1, y1, x2, y2)
        """
        indices = (bbox[0] * shape[1], bbox[1] * shape[0], bbox[2] * shape[1], bbox[3] * shape[0])
        return tuple(map(int, indices))
    
    def get_roi_slice(bbox: Tuple[float, float, float, float], tensor: torch.Tensor) -> torch.Tensor:
        """
        获取ROI的切片
        :param bbox: 边界框相对值 (x1, y1, x2, y2)
        :param tensor: 图像张量 (h, w, [c])
        :return: ROI的切片 (h, w, [c])
        """
        shape = tensor.shape[:2]
        idx = CustomSegmentor.bbox_to_indices(bbox, shape)
        return tensor[idx[1]:idx[3], idx[0]:idx[2]]

class DecisionTree(object):
    """
    决策树结构体
    """
    def __init__(self, tree: DecisionTreeClassifier, offset_features: torch.Tensor):
        self.sk_tree = tree
        self.features = offset_features     # 特征偏移向量 (feature_count, 4)
        self.features_valid = torch.any(self.features != 0, dim=1)  # 有效特征的掩码 (feature_count)

class RDFSegmentor(CustomSegmentor):
    """
    随机森林分割方法
    """
    def __init__(self, 
                 model_path: str,   # 模型路径
                 roi_detection: bool=True, # 是否使用ROI检测
                 roi_extended: float = 0.0, # ROI扩展比例
                 *,
                 pixel_count: int=2000,   # 每张图片采样的像素数量
                #  threshold_split: int=50,   # 深度特征的比较阈值的分割点数目
                #  threshold_min: float=-1.0,   # 深度特征的比较阈值的最小值
                #  threshold_max: float=1.0,   # 深度特征的比较阈值的最大值
                 offset_min: float=0.0,    # 偏移向量长度的最小值
                 offset_max: float=10.0,     # 偏移向量长度的最大值

                 sample_per_tree=1000,   # 每棵树训练的样本数量
                 max_depth=None,    # 树的最大深度
                 min_samples_split=2,  # 节点分裂所需的最小样本数
                 min_samples_leaf=1,   # 叶子节点最少包含的样本数
                 random_state=None,   # 随机数种子
                 feature_count=1000,   # 每棵树采样的特征数量
                 max_features=None,   # 特征选择的最大数量

                 tree_count=3,    # 随机森林的树的数量
                 ):
        super().__init__(model_path=model_path, roi_detection=roi_detection, roi_extended=roi_extended)

        # 深度特征采样参数
        self.pixel_count = pixel_count
        # self.thresholds: torch.Tensor[float] = torch.linspace(threshold_min, threshold_max, threshold_split, device=self.device, dtype=torch.float32)
        self.offset_min = offset_min
        self.offset_max = offset_max

        # 决策树训练参数
        self.sample_per_tree = sample_per_tree
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.feature_count = feature_count
        self.max_features = max_features

        # 随机森林参数
        self.tree_count = tree_count

    @staticmethod
    def get_depth_feature(depth_map: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        获取深度特征
        :param depth_map: 深度图 (w*h) [0, 1]
        :param u: 偏移向量1 (2)
        :param v: 偏移向量2 (2)
        :return: 深度特征 (w*h)
        """
        def get_depth_feature_single(depth_map: torch.Tensor, offset: torch.Tensor):
            offset_x = offset[0] / depth_map
            offset_y = offset[1] / depth_map
            # 构造索引矩阵
            indices_x = torch.arange(depth_map.shape[0], device=depth_map.device)
            indices_y = torch.arange(depth_map.shape[1], device=depth_map.device)
            grid_x, grid_y = torch.meshgrid(indices_x, indices_y, indexing='ij')
            # 计算偏移后的坐标 判断是否有效
            grid_x = grid_x + offset_x
            grid_y = grid_y + offset_y
            valid = (grid_x >= 0) & (grid_x < depth_map.shape[0]) & \
                    (grid_y >= 0) & (grid_y < depth_map.shape[1])

            # 计算深度特征 无效的深度值设置为最大值 1
            depth_feature = torch.ones_like(depth_map, dtype=depth_map.dtype, device=depth_map.device)
            depth_feature[valid] = depth_map[grid_x[valid].int(), grid_y[valid].int()]
            return depth_feature
        
        return get_depth_feature_single(depth_map, u) - get_depth_feature_single(depth_map, v)
    
    @staticmethod
    def get_multiple_depth_feature(depth_map: torch.Tensor, pixel_indices: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        获取深度特征
        :param depth_map: 深度图 (map_count, w, h) [0, 1]
        :param pixel_indices: 采样的像素索引 (2, pixel_count)
        :param u: 偏移向量1 (2, n)
        :param v: 偏移向量2 (2, n)
        :return: 深度特征 (map_count, pixel_count, n)
        """
        def get_depth_feature_single(depth_map: torch.Tensor, pixel_indices: torch.Tensor, offset: torch.Tensor):
            depth = depth_map[:, pixel_indices[0], pixel_indices[1]][None, :, :, None]
            norm_offset = offset[:, None, None, :] / depth    # (2, map_count, pixel_count, n)
            # 计算偏移后的坐标 判断是否有效
            grid = pixel_indices[:, None, :, None] + norm_offset
            valid = (grid[0] >= 0) & (grid[0] < depth_map.shape[0]) & \
                    (grid[1] >= 0) & (grid[1] < depth_map.shape[1])

            # 计算深度特征 无效的深度值设置为最大值 1
            depth_feature = torch.ones((depth_map.shape[0], pixel_indices.shape[1], offset.shape[1]), dtype=depth_map.dtype, device=depth_map.device)
            map_indices = torch.arange(depth_map.shape[0], device=depth_map.device)
            map_indices = map_indices[:, None, None].expand_as(depth_feature)
            depth_feature[valid] = depth_map[map_indices[valid], grid[0][valid].int(), grid[1][valid].int()]
            return depth_feature
        
        depth_map = depth_map[None, :, :] if len(depth_map.shape) == 2 else depth_map
        return get_depth_feature_single(depth_map, pixel_indices, u) - get_depth_feature_single(depth_map, pixel_indices, v)
 
    def train_tree(self, dataset: RHDDataset, 
                   sample_indices: List[int] | None = None,  
                   features: torch.Tensor | None = None) -> DecisionTree:
        """
        训练一棵决策树
        :param dataset: 数据集
        :param sample_indices: 样本索引 (默认随机选取)
        :param features: 已有的特征向量 (n, 4)
        :return: 决策树和实际使用的特征
        """
        self.buffer_grid_indices(dataset.image_shape)
        # 随机选取样本
        indices = np.random.choice(len(dataset), size=self.sample_per_tree, replace=False) if sample_indices is None else sample_indices
        samples: List[RHDDatasetItem] = [dataset[i] for i in indices]
        logging.debug(f"训练样本: {','.join(map(str, indices))}")

        # 获取特征向量
        if features is None:
            u = self.get_offset_vectors(self.feature_count)
            v = self.get_offset_vectors(self.feature_count)
        elif features.shape[0] < self.feature_count:
            u = self.get_offset_vectors(self.feature_count - features.shape[0])
            u = torch.cat([features[:, :2].transpose(0, 1), u], dim=1)
            v = self.get_offset_vectors(self.feature_count - features.shape[0])
            v = torch.cat([features[:, 2:].transpose(0, 1), v], dim=1)
        else:
            u = features[:, :2].transpose(0, 1)
            v = features[:, 2:].transpose(0, 1)

        # 获取抽样像素点
        pixel_indices = torch.randint(0, dataset.pixel_count, size=(2, self.pixel_count), device=self.device)
        pixel_indices[0] = pixel_indices[0] // dataset.image_shape[0]
        pixel_indices[1] = pixel_indices[1] % dataset.image_shape[1]

        depth_map = torch.zeros((self.sample_per_tree, dataset.image_shape[0], dataset.image_shape[1]), device=self.device)
        labels = np.zeros((self.pixel_count * self.sample_per_tree))
        for i in range(self.sample_per_tree):
            s = samples[i]
            # 记录深度图
            depth_map[i] = s.depth
            # 获取标签
            labels[i * self.pixel_count: (i + 1) * self.pixel_count] = s.mask[pixel_indices[0].cpu(), pixel_indices[1].cpu()]

        # 计算深度特征
        depth_features = RDFSegmentor.get_multiple_depth_feature(depth_map, pixel_indices, u, v)  # (image_count, pixel_count, n)
        depth_features = depth_features.reshape(-1, self.feature_count)

        # 训练决策树
        tree = DecisionTreeClassifier(max_depth=self.max_depth, 
                                      min_samples_leaf=self.min_samples_leaf, 
                                      min_samples_split=self.min_samples_split, 
                                      random_state=self.random_state,
                                      max_features=self.max_features
                                    )
        tree.fit(depth_features.cpu(), labels)
        # 记录实际使用的特征
        features = torch.zeros((self.feature_count, 4), device=self.device)
        for feature_idx in tree.tree_.feature:
            if feature_idx >= 0:
                features[feature_idx] = torch.concat((u[:, feature_idx], v[:, feature_idx]))
        return DecisionTree(tree, features)
    
    def train_forest(self, dataset: RHDDataset,) -> List[DecisionTree]:
        """
        训练随机森林
        :param dataset: 数据集
        :return: 决策树列表
        """
        self.buffer_grid_indices(dataset.image_shape)
        trees = []
        for _ in tqdm(range(self.tree_count), desc="训练决策树"):
            tree = self.train_tree(dataset)
            trees.append(tree)
        return trees

    def train_tree_iterative(self, dataset: RHDDataset, 
                             iter_count: int=10, 
                             test_count: int=10, 
                             keep_features: float | int=1.0) -> DecisionTree:
        """
        迭代地训练一棵决策树
        :param dataset: 数据集
        :param iter_count: 迭代次数
        :param test_count: 测试样本数量
        :param keep_features: 每轮保留的特征最大数量/比例
        :return: 决策树和实际使用的特征
        """
        self.buffer_grid_indices(dataset.image_shape)

        best_features = None
        best_tree = None
        best_iou = 0

        for i in range(iter_count):
            logging.info(f"开始第 {i+1} / {iter_count} 轮迭代训练")
            tree = self.train_tree(dataset, features=best_features)

            logging.info(f"开始第 {i+1} / {iter_count} 轮迭代评估")
            test_indices = np.random.choice(len(dataset), size=test_count, replace=False)
            test_samples: List[RHDDatasetItem] = [dataset[i] for i in test_indices]
            gt_mask = [sample.mask for sample in test_samples]
            
            depth_map = torch.stack([sample.depth for sample in test_samples])
            predict_mask = self.predict_mask_impl(depth_map, tree)
            test_iou = SegmentationEvaluation.mean_iou_static(predict_mask, gt_mask)

            if test_iou > best_iou:
                logging.info(f"第 {i+1} / {iter_count} 轮迭代更优: current_iou={test_iou:.4f} > best_iou={best_iou:.4f}")
                best_iou = test_iou
                best_tree = tree
                # 保留最优且最重要的特征
                if isinstance(keep_features, float):
                    keep_feature_count = int(keep_features * len(tree.sk_tree.tree_.feature))
                else:
                    keep_feature_count = min(keep_features, tree.features.shape[0])
                feature_indices = np.argsort(tree.sk_tree.feature_importances_)[::-1][:keep_feature_count]
                best_features = tree.features[feature_indices.copy(), :]
            else:
                logging.info(f"第 {i+1} / {iter_count} 轮迭代无效: current_iou={test_iou:.4f} <= best_iou={best_iou:.4f}")

        return best_tree
    
    def predict_mask(self, rgb_map: torch.Tensor, depth_map: torch.Tensor, return_prob: bool=False) -> np.ndarray:
        return self.predict_mask_impl(depth_map, None, return_prob)
    
    def predict_mask_impl(self, depth_map: torch.Tensor, tree_list: DecisionTree | List[DecisionTree], return_prob: bool=False) -> np.ndarray:
        """
        预测分类掩膜
        :param depth_map: 深度图 (map_count, w, h) [0, 1]
        :param tree: 决策树
        :param return_prob: 是否返回预测概率 否则返回众数（分类结果）
        :return: 分类掩膜 (map_count, w, h)
        """
        original_shape = depth_map.shape
        if len(depth_map.shape) == 2:
            depth_map = depth_map[None, :, :]
        map_count = depth_map.shape[0]
        pixel_count = depth_map.shape[1] * depth_map.shape[2]
        if isinstance(tree_list, DecisionTree):
            tree_list = [tree_list]
            
        self.buffer_grid_indices(depth_map[0].shape)

        ret = np.zeros((map_count, pixel_count, len(tree_list)), dtype=np.uint8)
        for i in range(len(tree_list)):
        # for i in tqdm(range(len(tree_list)), desc="预测分类掩膜"):
            # 计算深度特征
            tree = tree_list[i]
            u = tree.features[tree.features_valid, :2].transpose(0, 1)
            v = tree.features[tree.features_valid, 2:].transpose(0, 1)
            depth_features = RDFSegmentor.get_multiple_depth_feature(depth_map, self.flatten_indices, u, v)  # (map_count, w*h, n)
        
            # 进行预测
            features = torch.zeros(depth_map.numel(), tree.features.shape[0], device=self.device)
            features[:, tree.features_valid] = depth_features.view(-1, depth_features.shape[2])
            predicts = tree.sk_tree.predict(features.cpu())  # (map_count * w*h)
            ret[:, :, i] = predicts.reshape(map_count, pixel_count)

        if len(tree_list) == 1:
            return ret.reshape(original_shape).astype(np.float32 if return_prob else np.uint8)
        
        # 返回预测概率
        if return_prob:
            return (ret.astype(np.int32).sum(axis=2).astype(np.float32) / len(tree_list)).reshape(original_shape)
        # 返回众数
        else:
            mode, _ = stats.mode(ret, axis=2, keepdims=False)
            return mode.astype(np.uint8).reshape(original_shape)


    def get_offset_vectors(self, count: int) -> torch.Tensor:
        """
        获取一组偏移向量
        :param count: 向量数量
        :return: 偏移向量 (2, count)
        """
        length = torch.rand(count, device=self.device) * (self.offset_max - self.offset_min) + self.offset_min
        angle = torch.rand(count, device=self.device) * (2 * 3.141592653589793)
        x = length * torch.cos(angle)
        y = length * torch.sin(angle)
        return torch.stack([x, y], dim=0)
    
    
class ResNetSegmentor(CustomSegmentor):
    """
    基于ResNet的语义分割器
    """
    def __init__(self,
                 model_path: str,   # 模型路径
                 roi_detection: bool=True, # 是否使用ROI检测
                 roi_extended: float = 0.0, # ROI扩展比例
                 *,
                 predict_width: int=256,   # 预测输入图像宽度
                 predict_height: int=256,   # 预测输入图像高度
                 ):
        super().__init__(model_path=model_path, roi_detection=roi_detection, roi_extended=roi_extended)

        self.model = ResNet.get_model(ck_path=model_path)
        _ = self.model.eval()
        self.device = next(self.model.parameters()).device

        self.predict_width = predict_width
        self.predict_height = predict_height

    def predict_mask(self, rgb_map: torch.Tensor, depth_map: torch.Tensor, return_prob: bool=False) -> np.ndarray:
        if not self.roi_detection:
            pred, center = self.predict_mask_impl(rgb_map, depth_map, return_prob)
            return pred
        elif len(rgb_map.shape) == 4:
            pred_list = []
            for i in range(rgb_map.shape[0]):
                pred, center = self.predict_mask(rgb_map[i], depth_map[i], return_prob)
                pred_list.append(pred)
            return np.stack(pred_list, axis=0)
        else:
            bbox = self.get_roi_bbox(rgb_map)
            if bbox == None:
                pred, center = self.predict_mask_impl(rgb_map, depth_map, return_prob)
                return pred
            else:
                rgb_index = CustomSegmentor.bbox_to_indices(bbox, rgb_map.shape[-3:-1])
                rgb = CustomSegmentor.get_roi_slice(bbox, rgb_map)
                depth = CustomSegmentor.get_roi_slice(bbox, depth_map)
                pred, center = self.predict_mask_impl(rgb, depth, return_prob)
                ret = np.zeros(rgb_map.shape[:-1], dtype=pred.dtype)
                ret[rgb_index[1]:rgb_index[3], rgb_index[0]:rgb_index[2]] = pred
                return ret
            
    def predict_mask_impl(self, rgb_map: torch.Tensor, depth_map: torch.Tensor, return_prob: bool=False) -> np.ndarray:
        """
        预测掩码
        :param rgb_map: RGB图 (batch_size, h, w, 3) [0, 255]
        :param depth_map: 深度图 (batch_size, h, w) [0, 1]
        :param return_prob: 是否返回预测概率 否则返回众数（分类结果）
        :return: (分类掩膜 (batch_size, h, w), 中心点位置 (batch_size, 2) (y, x))
        """

        # 将图像处理为模型输入
        origin_shape = rgb_map.shape
        single_input = len(rgb_map.shape) == 3
        predict_map = rgb_map
        if single_input:
            predict_map = predict_map[None, :, :, :]
        predict_map = predict_map.to(self.device).permute(0, 3, 1, 2).float() / 255.0
        predict_map = transforms.Resize((self.predict_height, self.predict_width), antialias=False)(predict_map)
        predict_map = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(predict_map)

        # 进行预测
        prob: torch.Tensor = self.model(predict_map).detach()
        prob = F.softmax(prob, dim=1)[:, 1, :, :] # 手部掩膜概率

        # 还原尺寸
        prob = transforms.Resize((origin_shape[-3], origin_shape[-2]), antialias=False)(prob)
        mask = torch.zeros_like(prob, dtype=torch.uint8, device=self.device)
        mask[prob > 0.5] = 1

        # 计算中心点
        # indices = torch.nonzero(mask, as_tuple=False).float()
        center = torch.zeros((prob.shape[0], 2), device=self.device)
        # for i in range(center.shape[0]):
        #     i_mask = indices[:, 0] == i
        #     if i_mask.sum() == 0:
        #         continue
        #     center[i, 0] = indices[i_mask, 1].mean().item()
        #     center[i, 1] = indices[i_mask, 2].mean().item()

        # if single_input:
        #     depth_map = depth_map[None, :, :]
        # depth_map = transforms.Resize((origin_shape[-3], origin_shape[-2]), antialias=True)(depth_map)
        # center_depth = depth_map[torch.arange(depth_map.shape[0]), center[:, 0].long(), center[:, 1].long()]
        # center_depth = center_depth[:, None, None]
        # mask |= (depth_map > (center_depth - 0.005)) & (depth_map < (center_depth + 0.005))

        if single_input:
            prob = prob[0, :, :]
            mask = mask[0, :, :]
            center = center[0, :]
        prob = prob.cpu().numpy()
        mask = mask.cpu().numpy().astype(np.uint8)
        return prob if return_prob else mask, center


class SkinColorSegmentation(object):
    """
    肤色分割方法
    """
    def __init__(self, dataset: RHDDataset, config: Config):
        self.dataset = dataset
        self.config = config
        self.device = dataset.device
        self.model_path = f"{os.path.dirname(os.path.abspath(__file__))}/model/{self.config.dataset_name}.pickle"
        self.model = None

    def segment(self, image: RHDDatasetItem) -> List[Tuple[torch.Tensor, int]]:
        """
        对单张图片进行肤色分割
        :param image: 单张图片
        :return: 肤色分割结果 [(区域掩膜, 区域大小),...]
        """
        if self.model is None:
            self.load_trained_model()
        
        color = image.color
        rgb = image.color_rgb
        seed_mask = self.__get_uv_mask(color, self.model, self.config.seg_seed_threshold)
        regioin_mask = self.__get_uv_mask(color, self.model, self.config.seg_blob_threshold)
        seed_indices = seed_mask.nonzero(as_tuple=False)

        # 遍历所有种子点 进行区域生长
        regions = []
        for i in range(seed_indices.shape[0]):
            seed = seed_indices[i]
            region = self.__grow_region(seed, regioin_mask)
            # 去除已经被包含的区域
            seed_mask[region] = False
            regioin_mask[region] = False
            # 去除过小的区域
            region_size = region.count_nonzero()
            if region_size < self.config.seg_min_region_size:
                continue
            # 利用卷积扩展区域
            extend_radius = 5
            kernal = torch.ones((1, 1, extend_radius * 2 + 1, extend_radius * 2 + 1), dtype=torch.float32, device=self.device, requires_grad=False)
            region = F.conv2d(region.float()[None, None, :, :], kernal, padding=extend_radius).squeeze() > 0
            regions.append((region, region_size))

        # 按区域大小排序
        regions.sort(key=lambda x: x[1], reverse=True)

        region_count = len(regions)
        plt.subplot(1, region_count + 1, 1)
        plt.imshow(rgb)
        for i in range(region_count):
            plt.subplot(1, region_count + 1, i + 2)
            plt.imshow(regions[i][0].cpu().numpy())

        plt.show()
        return regions

    def __get_uv_mask(self, yuv: torch.Tensor, uv_model: torch.Tensor, threshold: float) -> torch.Tensor:
        u = yuv[:, :, 1].flatten().int()
        v = yuv[:, :, 2].flatten().int()
        value = uv_model[u, v]
        mask = value > threshold
        return mask.view(yuv.shape[0], yuv.shape[1])
    
    def __grow_region(self, seed: torch.Tensor, region_mask: torch.Tensor) -> torch.Tensor:
        """
        区域生长算法
        :param seed: 种子点 (2)
        :param mask: 可生长区域的掩膜 (w*h)
        :return: 生长后的区域 (w*h)
        """
        region = torch.zeros_like(region_mask, dtype=torch.bool, requires_grad=False)
        region[seed[0], seed[1]] = 1.0
        pre_region = torch.zeros_like(region_mask, dtype=torch.bool, requires_grad=False)
        kernal = torch.tensor([[1, 1, 1], 
                               [1, 0, 1], 
                               [1, 1, 1]], dtype=torch.float32, device=self.device, requires_grad=False)
        kernal = kernal[None, None, :, :]
        
        # count = 0
        while not torch.equal(region, pre_region):
            pre_region[:, :] = region
            # 利用卷积扩展区域
            region = F.conv2d(region.float()[None, None, :, :], kernal, padding=1).squeeze() > 0
            region &= region_mask
            # count += 1
            # if count % 10 == 0:
            #     plt.imshow(region.float().cpu().numpy())
                # plt.show()
        return region

    def load_trained_model(self, max_size=-1, force_retrain=False) -> torch.Tensor:
        """
        加载训练好的模型
        :param max_size: 最大训练集大小，-1表示使用全部数据
        :param force_retrain: 强制重新训练模型
        """
        if not self.__have_trained_model() or force_retrain:
            self.model = self.train_model(max_size)
            if not os.path.exists(os.path.dirname(self.model_path)):
                os.makedirs(os.path.dirname(self.model_path))
            torch.save(self.model, self.model_path)
        else:
            self.model = torch.load(self.model_path)
        return self.model

    def train_model(self, max_size=-1) -> torch.Tensor:
        """
        训练模型
        :param max_size: 最大训练集大小，-1表示使用全部数据
        """
        sample_count = len(self.dataset) if max_size == -1 or max_size > len(self.dataset) else max_size
        image_size = self.dataset.image_shape[0] * self.dataset.image_shape[1]
        ps = 0      # 皮肤色的频率
        pc: torch.Tensor = torch.zeros((65536,), dtype=torch.int32, device=self.device)      # 各种颜色出现的频率 按UV空间分割 (65536)
        pcs: torch.Tensor = torch.zeros((65536,), dtype=torch.int32, device=self.device)     # 各种颜色在皮肤色中出现的频率 按UV空间分割 (65536)
        
        print("开始训练模型...")
        with torch.no_grad():
            for i in tqdm(range(sample_count)):
                sample = self.dataset[i]
                u = sample.color[:, :, 1].flatten().int()
                v = sample.color[:, :, 2].flatten().int()
                uv = u * 256 + v
                hand_mask = sample.origin_mask.flatten() >= 2      # 手掌掩膜 分类>=2为手掌的部分

                ps_cur = hand_mask.sum(where=hand_mask)
                
                pc_cur = torch.bincount(uv, minlength=pc.numel())
                pcs_cur = torch.bincount(uv[hand_mask], minlength=pcs.numel())
                ps += ps_cur
                pc += pc_cur
                pcs += pcs_cur

        # 频率转换为概率
        ps = ps / (sample_count * image_size)
        pc = pc.float() / (sample_count * image_size)
        pcs = pcs.float() / (ps * sample_count * image_size)
        print(f"训练完成 皮肤色概率: {ps:.6f}")
        print(f"各种颜色出现的概率之和: {float(pc.sum()):.6f}")
        print(f"各种颜色在皮肤色中出现的概率之和: {float(pcs.sum()):.6f}")

        # 计算一个颜色是皮肤色的概率P(S|C)
        result: torch.Tensor = torch.zeros((65536,), dtype=torch.float32, device=self.device)
        valid_mask = pc > 0
        result[valid_mask] = pcs[valid_mask] * ps / pc[valid_mask]
        return result.view(256, 256)

    def __have_trained_model(self) -> bool:
        return os.path.exists(self.model_path)
