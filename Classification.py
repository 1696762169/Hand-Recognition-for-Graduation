import struct
from typing import Tuple

import torch
import numpy as np
import cv2
from fastdtw import fastdtw

# 导入卡方系数
from scipy.stats import chisquare

class DTWClassifier(object):
    """
    动态时间规整（DTW）分类器
    """
    def __init__(self,
                 *,
                 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 ):
        self.device = device

    def get_distance(self, x_features: torch.Tensor, y_features: torch.Tensor) -> Tuple[float, np.ndarray]:
        """
        计算两个序列的距离
        :param x_features: 序列1的特征 (x_point_count, feature_count)
        :param y_features: 序列2的特征 (y_point_count, feature_count)
        :return: (DTW距离值, DTW路径)
        """

        distance, path = fastdtw(x_features.cpu().numpy(), y_features.cpu().numpy(), dist=self.__get_point_distance)
        path_ret = np.array(path).T
        return distance, path_ret

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
    
    def from_file(file_path: str, sample_idx: int) -> Tuple[np.ndarray, int, int]:
        """
        从文件中解析特征
        :param file_path: 文件路径
        :param sample_idx: 样本索引
        :return: (特征, 类别索引)
        """
        with open(file_path, 'rb') as f:
            sample_count = struct.unpack('I', f.read(4))[0]
            sample_pos = np.frombuffer(f.read(sample_count * 4), dtype=np.int32)
            f.seek((sample_idx + 1) * 4)
            sample_seek = struct.unpack('i', f.read(4))[0]
            f.seek(sample_seek)
            byte_count = struct.unpack('I', f.read(4))[0]
            data = f.read(byte_count)
            return DTWClassifier.from_byte(data)
        