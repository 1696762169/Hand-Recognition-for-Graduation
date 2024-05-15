import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
import logging

from Dataset.RHD import RHDDataset, RHDDatasetItem
from Config import Config
from matplotlib import pyplot as plt
from typing import List, Tuple

def segment_image(image, config):
    pass

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
    

class DecisionTree(object):
    """
    决策树结构体
    """
    def __init__(self, tree: DecisionTreeClassifier, offset_features: torch.Tensor):
        self.sk_tree = tree
        self.features = offset_features     # 特征偏移向量 (feature_count, 4)
        self.features_valid = torch.any(self.features != 0, dim=1)  # 有效特征的掩码 (feature_count)

class RDFSegmentor(object):
    """
    随机森林分割方法
    """
    def __init__(self, *,
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
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

        # 网格索引缓存
        self.grid_indices: torch.Tensor = None  # (2, w*h)

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
 
    def train_tree(self, dataset: RHDDataset, sample_indices: List[int] | None = None) -> DecisionTree:
        """
        训练一棵决策树
        :param dataset: 数据集
        :param sample_indices: 样本索引 (默认随机选取)
        :return: 决策树和实际使用的特征
        """
        self.buffer_grid_indices(dataset.image_shape)
        # 随机选取样本
        indices = np.random.choice(len(dataset), size=self.sample_per_tree, replace=False) if sample_indices is None else sample_indices
        samples: List[RHDDatasetItem] = [dataset[i] for i in indices]

        # 获取特征向量
        u = self.get_offset_vectors(self.feature_count)
        v = self.get_offset_vectors(self.feature_count)

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
                             iter_count: int, 
                             test_count: int, 
                             keep_features: float | int=10) -> DecisionTree:
        """
        迭代地训练一棵决策树
        :param dataset: 数据集
        :param iter_count: 迭代次数
        :param test_count: 测试样本数量
        :param keep_features: 每轮保留的特征最大数量/比例
        :return: 决策树和实际使用的特征
        """
        self.buffer_grid_indices(dataset.image_shape)
        # 随机选取样本
        indices = np.random.choice(len(dataset), size=self.sample_per_tree, replace=False)
        samples: List[RHDDatasetItem] = [dataset[i] for i in indices]

        best_tree = None
        best_iou = 0

        for i in range(iter_count):
            logging.info(f"开始第{i+1}/{iter_count}轮迭代训练")
            tree = self.train_tree(dataset)
            logging.info(f"开始第{i+1}/{iter_count}轮迭代评估")


    def predict_mask(self, depth_map: torch.Tensor, tree_list: DecisionTree | List[DecisionTree], return_prob: bool=False) -> np.ndarray:
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
            depth_features = RDFSegmentor.get_multiple_depth_feature(depth_map, self.grid_indices, u, v)  # (map_count, w*h, n)
        
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
    
    def buffer_grid_indices(self, shape: Tuple[int, int]) -> None:
        """
        计算并缓存网格索引
        :param shape: 图像的尺寸 (w, h)
        """
        if self.grid_indices is not None and self.grid_indices.shape[1:] == shape:
            return
        indices_x = torch.arange(shape[0], device=self.device)
        indices_y = torch.arange(shape[1], device=self.device)
        self.grid_x, self.grid_y = torch.meshgrid(indices_x, indices_y, indexing='ij')
        self.grid_indices = torch.stack([self.grid_x.flatten(), self.grid_y.flatten()], dim=0)