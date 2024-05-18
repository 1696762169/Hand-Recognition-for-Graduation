import cv2
import torch
from scipy import ndimage
from torch.nn import functional as F
import numpy as np

from matplotlib import pyplot as plt

class SobelOperator(torch.nn.Module):
    sobel_kernel_x = torch.tensor([[1, 0, -1], 
                                   [2, 0, -2], 
                                   [1, 0, -1]], 
                                   dtype=torch.float32)[None, None, :, :]
    sobel_kernel_y = torch.tensor([[1, 2, 1], 
                                   [0, 0, 0], 
                                   [-1, -2, -1]], 
                                   dtype=torch.float32)[None, None, :, :]
    
    def __init__(self, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(SobelOperator, self).__init__()
        # 初始化 Sobel 核
        self.weight_x = torch.nn.Parameter(SobelOperator.sobel_kernel_x.to(device))
        self.weight_y = torch.nn.Parameter(SobelOperator.sobel_kernel_y.to(device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算图像的梯度幅度
        :param x: 输入图像 (B, C, H, W)
        :return: 图像的梯度幅度 (B, C, H, W)
        """

        # 应用 Sobel 核
        out_x = F.conv2d(x, self.weight_x, padding=1)
        out_y = F.conv2d(x, self.weight_y, padding=1)
        # 计算梯度幅度
        magnitude = F.relu(out_x**2 + out_y**2).sqrt()
        # 归一化
        max_val = torch.amax(magnitude, dim=(1, 2, 3))[:, None, None, None]
        magnitude = magnitude / (max_val + 1e-12)
        return magnitude
    

class ThreeDSCTracker:
    """
    3D-SC描述符特征生产器
    """
    def __init__(self,
                 *,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.device = device

    def extract_contour(self, depth_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        提取轮廓特征
        :param depth_map: 深度图 (B, H, W) [0, 1]
        :param mask: 预测手部掩码图 (B, H, W) 0/1
        :return: 轮廓特征 (B, H, W)
        """
        origin_shape = depth_map.shape
        single_input = len(origin_shape) == 2
        batch_size = 1 if single_input else origin_shape[0]
        depth_map = (depth_map[None, :, :] if single_input else depth_map).float().to(self.device)
        mask = (mask[None, :, :] if single_input else mask).to(self.device)
        # mask[(depth_map <= 1e-6) | (depth_map >= 1 - 1e-6)] = 0     # 过滤一些明显的非法值
        mask_sum = torch.sum(mask, dim=(1, 2))

        # 计算梯度幅度
        sobel_op = SobelOperator(self.device)
        # grad_magnitude: torch.Tensor = sobel_op(depth_map[:, None, :, :])
        # grad_magnitude = grad_magnitude.detach()

        # 原始掩码图仅考虑最大的区域
        for i in range(batch_size):
            n_label, labels, stats, centroids = cv2.connectedComponentsWithStats(mask[i].cpu().numpy().astype(np.uint8), connectivity=8)
            largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
            mask[i] = torch.from_numpy(labels == largest_label).byte().to(mask.device)
        
        # 计算手部深度均值 与 方差 并排除深度比较远的点
        depth_mean = (torch.sum(depth_map * mask, dim=(1, 2)) / mask_sum)
        threshold = torch.zeros((batch_size), device=self.device)
        for i in range(batch_size):
            depth_mean[i] = torch.quantile(depth_map[i][mask[i] == 1], 0.5)
            threshold[i] = depth_mean[i] - torch.quantile(depth_map[i][mask[i] == 1], 0.1)
            # plt.hist(depth_map[i][mask[i] == 1].cpu().numpy(), bins=100)
            # plt.axvline(x=threshold[i].cpu().numpy(), color='r')
            # plt.show()
        threshold, depth_mean = threshold[:, None, None], depth_mean[:, None, None]
        # depth_std = torch.sqrt(torch.sum((depth_map - depth_mean)**2 * mask, dim=(1, 2)) / mask_sum)
        # threshold = depth_std[:, None, None] * 0.5
        depth_mask = mask == 1
        depth_mask |= (depth_map > depth_mean - 3 * threshold) & (depth_map < depth_mean + 3 * threshold)
        depth_mask &= (depth_map > depth_mean - 5 * threshold) & (depth_map < depth_mean + 5 * threshold)

        # 计算后的掩码图也仅考虑最大的区域
        for i in range(batch_size):
            n_label, labels, stats, centroids = cv2.connectedComponentsWithStats(depth_mask[i].cpu().numpy().astype(np.uint8), connectivity=8)
            largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
            depth_mask[i] = torch.from_numpy(labels == largest_label).bool().to(depth_mask.device)
        masked_depth_map = depth_map * depth_mask.float()

        # 形态学开运算
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        masked_depth_map = cv2.morphologyEx(masked_depth_map.cpu().numpy(), cv2.MORPH_OPEN, kernel)
        masked_depth_map = torch.from_numpy(masked_depth_map).to(self.device)

        # 计算梯度
        grad_magnitude: torch.Tensor = sobel_op(masked_depth_map[:, None, :, :])
        grad_magnitude = grad_magnitude.detach()

        # return depth_mask.float()[0, :, :] if single_input else depth_mask.float()
        # return masked_depth_map[0, :, :] if single_input else masked_depth_map
        return grad_magnitude[0, 0, :, :] if single_input else grad_magnitude[:, 0, :, :]