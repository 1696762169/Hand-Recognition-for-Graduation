import torch
import numpy as np
from torch import Tensor as tensor

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