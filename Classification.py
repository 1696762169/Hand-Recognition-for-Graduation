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
