import torch
import numpy as np
from torch import Tensor as tensor
from typing import List, Tuple

from Config import Config
from Dataset.RHD import RHDDataset, RHDDatasetItem
from HandModel import HandModel

class Evaluation:
    def __init__(self, config: Config, dataset: RHDDataset, models: list[HandModel]):
        self.config = config
        self.dataset = dataset
        self.models = models
        self.model_num = len(models)

    def evaluate(self, tensor: tensor) -> tensor:
        assert tensor.shape[0] == self.model_num, "参数数量与模型数量不匹配"

        ret = np.empty((self.model_num), dtype=float)
        for i in range(self.model_num):
            model = HandModel(self.models[i])
            model.set_pose(tensor[i])   # 设置姿态
            depth_h =model.render()     # 调用render函数渲染深度图
            ret[i] = self.__evaluate_one_model(self.dataset[i], torch.from_numpy(depth_h[:, :, 0] / 256.0))   # 计算评估值
            
        return torch.from_numpy(ret)
    
    def __evaluate_one_model(self, sample: RHDDatasetItem, depth_h: tensor) -> float:
        """
        计算待优化的估计值
        参考论文 https://hgpu.org/?p=6897 2.3节
        :param sample: 样本
        :param depth_h: 深度图 (H, W) 范围 [0, 1]
        :return: 评估值
        """

        # D(O, h, C)
        depth_diff = torch.min(torch.abs(depth_h - sample.depth), self.config.depth_max_diff)
        depth_match = (depth_diff < self.config.depth_match_threshold)

class ClassificationEvaluation(object):
    """
    分类评估(准确率, 精确率, 召回率, F1-score)
    """
    def __init__(self, class_num: int, predict_label: List[int], gt_label: List[int]):
        self.class_num = class_num
        self.predict_label = predict_label
        self.gt_label = gt_label
        assert len(predict_label) == len(gt_label), "预测标签数量与真实标签数量不匹配"
        self.sample_num = len(predict_label)

        # 计算混淆矩阵
        self.confusion_matrix = ClassificationEvaluation.__get_confusion_matrix(self.class_num, predict_label, gt_label)
        
    # 计算混淆矩阵
    @staticmethod
    def __get_confusion_matrix(class_num: int, predict_label: List[int], gt_label: List[int]) -> np.ndarray:
        confusion_matrix = np.zeros((class_num, class_num), dtype=int)
        for i in range(len(predict_label)):
            confusion_matrix[predict_label[i], gt_label[i]] += 1
        return confusion_matrix
    
    # 计算准确率
    def accuracy(self) -> float:
        return np.trace(self.confusion_matrix) / self.sample_num
    @staticmethod
    def accuracy(class_num: int, predict_label: List[int], gt_label: List[int]) -> float:
        return ClassificationEvaluation(class_num, predict_label, gt_label).accuracy()

    # 计算精确率
    def precision(self, class_id: int) -> float:
        return self.confusion_matrix[class_id, class_id] / np.sum(self.confusion_matrix[:, class_id])
    @staticmethod
    def precision(class_num: int, predict_label: List[int], gt_label: List[int], class_id: int) -> float:
        return ClassificationEvaluation(class_num, predict_label, gt_label).precision(class_id)
    
    # 计算召回率
    def recall(self, class_id: int) -> float:
        return self.confusion_matrix[class_id, class_id] / np.sum(self.confusion_matrix[class_id, :])
    @staticmethod
    def recall(class_num: int, predict_label: List[int], gt_label: List[int], class_id: int) -> float:
        return ClassificationEvaluation(class_num, predict_label, gt_label).recall(class_id)
    
    # 计算F1-score
    def f1_score(self, class_id: int) -> float:
        precision = self.precision(class_id)
        recall = self.recall(class_id)
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    @staticmethod
    def f1_score(class_num: int, predict_label: List[int], gt_label: List[int], class_id: int) -> float:
        return ClassificationEvaluation(class_num, predict_label, gt_label).f1_score(class_id)
    
class SegmentationEvaluation(object):
    """
    分割评估(IoU)
    """
    def __init__(self, predict_mask: List[np.ndarray], gt_mask: List[np.ndarray]):
        """
        :param predict_mask: 预测掩膜列表
        :param gt_mask: 真实掩膜列表
        :param save_matrix: 是否保存IoU矩阵
        """
        self.predict_mask = predict_mask
        self.gt_mask = gt_mask
        assert len(predict_mask) == len(gt_mask), "预测掩膜数量与真实掩膜数量不匹配"

        self.sample_num = len(predict_mask)

        # 计算IoU列表
        self.iou_list = SegmentationEvaluation.__get_iou_list(predict_mask, gt_mask)
        
    # 计算IoU列表
    @staticmethod
    def __get_iou_list(predict_mask: List[np.ndarray], gt_mask: List[np.ndarray]) -> List[float]:
        iou_list = []
        for i in range(len(predict_mask)):
            intersection = np.sum(np.logical_and(predict_mask[i], gt_mask[i]))
            union = np.sum(np.logical_or(predict_mask[i], gt_mask[i]))
            iou_list.append(intersection / union if union > 0 else 0.0)
        return iou_list
    
    # 计算平均IoU
    def mean_iou(self) -> float:
        return np.mean(self.iou_list)
    @staticmethod
    def mean_iou(predict_mask: List[np.ndarray], gt_mask: List[np.ndarray]) -> float:
        return SegmentationEvaluation(predict_mask, gt_mask).mean_iou()