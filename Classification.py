import time
import struct
import numbers
from collections import defaultdict
from typing import Tuple, List, Literal

import torch
import numpy as np
import cv2
from fastdtw import fastdtw, dtw
import tslearn.metrics as ts
import torch.nn.functional as F

from tqdm import tqdm

class FastDTW(object):
    """
    优化的快速 DTW 算法实现
    """
    dist_timer: float = 0
    min_timer: float = 0

    @staticmethod
    def fastdtw(x: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor, radius: int = 1, dist=None) -> Tuple[float, List[Tuple[int, int]]]:
        """
        使用FastDTW算法计算两个序列的 DTW 距离
        :param x: 序列 1
        :param y: 序列 2
        :param radius: 扩展窗口半径
        :param dist: 距离函数，默认为欧氏距离
        :return: 距离值和路径
        """
        FastDTW.dist_timer = 0
        FastDTW.min_timer = 0
        x, y, dist = FastDTW.__prep_inputs(x, y, dist)
        return FastDTW.__fastdtw_gpu(x, y, radius, dist)

    @staticmethod
    def dtw(x, y, dist=None):
        """
        计算两个序列的 DTW 距离
        :param x: 序列 1
        :param y: 序列 2窗口半径
        :param dist: 距离函数，默认为欧氏距离
        :return: 距离值和路径
        """
        FastDTW.dist_timer = 0
        FastDTW.min_timer = 0
        x, y, dist = FastDTW.__prep_inputs(x, y, dist)
        return FastDTW.__dtw(x, y, dist)
    

    @staticmethod
    def __prep_inputs(x: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor, dist):
        """
        处理输入数据为适合 DTW 计算的格式
        """
        if x.ndim == y.ndim > 1 and x.shape[1] != y.shape[1]:
            raise ValueError('second dimension of x and y must be the same')
        if isinstance(dist, numbers.Number) and dist <= 0:
            raise ValueError('dist cannot be a negative integer')

        if dist is None:
            if x.ndim == 1:
                dist = FastDTW.__difference
            else: 
                dist = FastDTW.__norm(p=1)
        elif isinstance(dist, torch.Tensor) or isinstance(dist, np.ndarray):
            if dist.shape != (x.shape[0], y.shape[0]):
                raise ValueError('dist shape must be (x.shape[0], y.shape[0])')
            dist_map = dist
            dist = lambda a, b: dist_map[a, b]
        elif isinstance(dist, numbers.Number):
            dist = FastDTW.__norm(p=dist)

        return x, y, dist

    @staticmethod
    def __fastdtw_gpu(x: torch.Tensor, y: torch.Tensor, radius: int, dist) -> Tuple[float, List[Tuple[int, int]]]:
        min_time_size = radius + 2
        len_x, len_y = x.shape[0], y.shape[0]
        if len_x < min_time_size or len_y < min_time_size:
            return FastDTW.__dtw(x, y, dist)

        x_shrinked = FastDTW.__reduce_by_half(x)
        y_shrinked = FastDTW.__reduce_by_half(y)

        distance, path = FastDTW.__fastdtw_gpu(x_shrinked, y_shrinked, radius=radius, dist=dist)
        window = FastDTW.__expand_window(path, len_x, len_y, radius)
        return FastDTW.__dtw(x, y, dist, window)

    @staticmethod
    def __dtw(x: torch.Tensor | np.ndarray, y: torch.Tensor | np.ndarray, dist, window: np.ndarray | None=None) -> Tuple[float, List[Tuple[int, int]]]:
        """
        实现DTW算法
        :param x: 序列 1
        :param y: 序列 2
        :param window: 计算窗口 (window_size, 2) 默认为None表示使用全部数据
        :param dist: 距离函数
        :return: 距离值和路径
        """
        len_x, len_y = x.shape[0], y.shape[0]
        if window is None:
            window = np.stack(np.nonzero(np.ones((len_x, len_y), dtype=np.byte)), axis=1)
        window_next = window.copy() + 1
        
        D = np.ones((len_x + 1, len_y + 1), dtype=np.float32) * float('inf')
        D[0, 0] = 0.0
        record = np.zeros((len_x + 1, len_y + 1, 2), dtype=np.int32)
        record[0, 0] = (0, 0)

        for pos in range(window.shape[0]):
            i, j = window[pos]
            i_next, j_next = window_next[pos]
            timer = time.perf_counter()
            temp = min((D[i, j_next], i, j_next), 
                    (D[i_next, j], i_next, j),
                    (D[i, j], i, j), 
                    key = lambda a: a[0])
            FastDTW.min_timer += time.perf_counter() - timer

            timer = time.perf_counter()
            dt = dist(x[i], y[j])
            FastDTW.dist_timer += time.perf_counter() - timer

            D[i_next, j_next] = dt + temp[0]
            record[i_next, j_next] = temp[1:]
                
        path = []
        i, j = len_x, len_y
        while not (i == j == 0):
            path.append((i - 1, j - 1))
            i, j = record[i, j, 0], record[i, j, 1]
        path.reverse()
        return (D[len_x, len_y], path)


    @staticmethod
    def __reduce_by_half(x: np.ndarray) ->  np.ndarray:
        return cv2.resize(x, (x.shape[1] // 2, x.shape[0] // 2), interpolation=cv2.INTER_LINEAR)
    def __reduce_by_half_gpu(x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x[:, :, None].permute(2, 1, 0), scale_factor=0.5, mode='linear', align_corners=False)
        return x[0].permute(1, 0)

    @staticmethod
    def __expand_window(path, len_x, len_y, radius):
        path_ = set(path)
        for i, j in path:
            for a, b in ((i + a, j + b)
                        for a in range(-radius, radius+1)
                        for b in range(-radius, radius+1)):
                path_.add((a, b))

        window_ = set()
        for i, j in path_:
            for a, b in ((i * 2, j * 2), (i * 2, j * 2 + 1),
                        (i * 2 + 1, j * 2), (i * 2 + 1, j * 2 + 1)):
                window_.add((a, b))

        window = []
        start_j = 0
        for i in range(0, len_x):
            new_start_j = None
            for j in range(start_j, len_y):
                if (i, j) in window_:
                    window.append((i, j))
                    if new_start_j is None:
                        new_start_j = j
                elif new_start_j is not None:
                    break
            start_j = new_start_j

        return np.array(window)
    
    @staticmethod
    def __difference(a, b):
        return abs(a - b)
    @staticmethod
    def __norm(p):
        return lambda a, b: np.linalg.norm(np.atleast_1d(a) - np.atleast_1d(b), p)


class DTWClassifier(object):
    """
    动态时间规整（DTW）分类器
    """
    def __init__(self,
                 *,
                 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 dtw_type: Literal['fastdtw', 'tslearn', 'custom'] = 'tslearn',
                 ):
        self.device = device
        self.dtw_type = dtw_type

    def get_distance(self, x_features: np.ndarray, y_features: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        计算两个序列的距离
        :param x_features: 序列1的特征 (x_point_count, feature_count)
        :param y_features: 序列2的特征 (y_point_count, feature_count)
        :return: (DTW距离值, DTW路径)
        """

        if self.dtw_type == 'tslearn':
            # path, distance = ts.dtw_path_from_metric(x_features, y_features, metric=DTWClassifier.__get_point_distance)
            path, distance = ts.dtw_path(x_features, y_features)
            return distance, np.array(path)
        elif self.dtw_type == 'fastdtw':
            distance, path = dtw(x_features, y_features, dist=DTWClassifier.__get_point_distance)
            path_ret = np.array(path).T
            return distance, path_ret
        elif self.dtw_type == 'custom':
            grid_x, grid_y = np.meshgrid(np.arange(x_features.shape[0]), np.arange(y_features.shape[0]), indexing='ij')
            x_points, y_points = x_features[grid_x.flatten()], y_features[grid_y.flatten()]
            dist: np.ndarray = 0.5 * np.sum((x_points - y_points) ** 2 / (x_points + y_points + 1e-6), axis=1)
            dist = dist.reshape(x_features.shape[0], y_features.shape[0])

            distance, path = FastDTW.dtw(np.arange(x_features.shape[0]), np.arange(y_features.shape[0]), dist=dist)
            path_ret = np.array(path).T
            return distance, path_ret
        else:
            raise ValueError(f'不支持的DTW实现类型: {self.dtw_type}')

    def get_distance_gpu(self, x_features: torch.Tensor, y_features: torch.Tensor) -> Tuple[float, np.ndarray]:
        if self.dtw_type == 'custom':
            grid_x, grid_y = torch.meshgrid(torch.arange(x_features.shape[0]), torch.arange(y_features.shape[0]), indexing='ij')
            x_points, y_points = x_features[grid_x.flatten()], y_features[grid_y.flatten()]
            dist = 0.5 * torch.sum((x_points - y_points) ** 2 / (x_points + y_points + 1e-6), dim=1)
            dist = dist.reshape(x_features.shape[0], y_features.shape[0]).cpu().numpy()

            distance, path = FastDTW.dtw(np.arange(x_features.shape[0]), np.arange(y_features.shape[0]), dist=dist)
            path_ret = np.array(path).T
            return distance, path_ret
        if self.dtw_type == 'tslearn':
            path, distance = ts.dtw_path_from_metric(x_features, y_features, metric=DTWClassifier.__get_point_distance_gpu)
            # distance = ts.dtw(x_features, y_features)
            return distance, None
        else:
            return self.get_distance(x_features.cpu().numpy(), y_features.cpu().numpy())
    
    @staticmethod
    def __get_point_distance(x_point: np.ndarray, y_point: np.ndarray) -> float:
        """
        计算两点之间的距离
        :param x_point: 点1的特征 (feature_count)
        :param y_point: 点2的特征 (feature_count)
        :return: 两点之间的距离
        """
        return 0.5 * np.sum((x_point - y_point) ** 2 / (x_point + y_point + 1e-6))
    @staticmethod
    def __get_point_distance_gpu(x_point: torch.Tensor, y_point: torch.Tensor) -> float:
        return (0.5 * torch.sum((x_point - y_point) ** 2 / (x_point + y_point + 1e-6))).item()

    @staticmethod
    def to_byte(features: np.ndarray, sample_idx: int, cls: int) -> bytes:
        """
        将特征转换为字节数据
        :param features: 特征 (point_count, feature_count)
        :param sample_idx: 样本索引
        :param cls: 类别索引
        :return: 字节数据
        """
        # 转换为字节数组
        features_bytes = features.astype(np.float32).tobytes()
        point_count = features.shape[0]
        return struct.pack('III', sample_idx, cls, point_count) + features_bytes
    @staticmethod
    def from_byte(data: bytes) -> Tuple[np.ndarray, int, int]:
        """
        从字节数据中解析特征
        :param data: 字节数据
        :return: (特征, 样本索引, 类别索引)
        """
        sample_idx, cls, point_count = struct.unpack('III', data[:12])
        features = np.frombuffer(data[12:], dtype=np.float32).reshape(point_count, -1)
        return features, sample_idx, cls
    @staticmethod
    def from_file(file_path: str, sample_idx: int) -> Tuple[np.ndarray, int, int]:
        """
        从文件中解析特征
        :param file_path: 文件路径
        :param sample_idx: 样本索引
        :return: (特征, 类别索引)
        """
        with open(file_path, 'rb') as f:
            f.seek((sample_idx + 1) * 4)
            sample_seek = struct.unpack('I', f.read(4))[0]
            f.seek(sample_seek)
            byte_count = struct.unpack('I', f.read(4))[0]
            data = f.read(byte_count)
            return DTWClassifier.from_byte(data)
    @staticmethod
    def from_file_all(file_path: str) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
        """
        从文件中解析所有特征
        :param file_path: 文件路径
        :return: (特征, 样本索引, 类别索引)
        """
        with open(file_path, 'rb') as f:
            sample_count = struct.unpack('I', f.read(4))[0]
            f.seek(4 + 4 * sample_count)
            features = []
            sample_idxs = np.empty(sample_count, dtype=np.uint32)
            cls_idxs = np.empty(sample_count, dtype=np.uint32)
            for i in tqdm(range(sample_count), desc='加载3D-SC特征'):
                byte_count = struct.unpack('I', f.read(4))[0]
                data = f.read(byte_count)
                feature, sample_idxs[i], cls_idxs[i] = DTWClassifier.from_byte(data)
                features.append(feature)

        return features, sample_idxs, cls_idxs