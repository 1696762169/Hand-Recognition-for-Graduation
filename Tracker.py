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
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 contour_threshold=0.5,     # 应用sobel算子后被认为是轮廓像素的阈值
                 contour_max_points=100,  # 轮廓最大点数
                 coutour_simplify_eplision=0.0005,  # 轮廓简化精度 （与轮廓长度相乘）
                 use_naive_simplify=False,  # 是否使用简单的轮廓简化算法
                 ):
        self.device = device

        # 轮廓简化参数
        self.contour_threshold = contour_threshold
        self.contour_max_points = contour_max_points
        self.coutour_simplify_eplision = coutour_simplify_eplision
        self.use_naive_simplify = use_naive_simplify

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
        or_scale, and_scale = 1.0, 2.0
        depth_mask |= (depth_map > depth_mean - or_scale * threshold) & (depth_map < depth_mean + or_scale * threshold)
        # depth_mask &= (depth_map > depth_mean - and_scale * threshold) & (depth_map < depth_mean + and_scale * threshold)

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
    
    def simplify_contour(self, contour: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """
        简化轮廓
        :param contour: 轮廓特征 (H, W) [0, 1]
        :return: 简化后的轮廓index (point_count, 2)
        """
        is_tensor = isinstance(contour, torch.Tensor)
        device = contour.device if is_tensor else torch.device('cpu')
        contour: np.ndarray = contour.cpu().numpy() if is_tensor else contour

        # 获取contour中值为1的点的坐标
        contour[contour < self.contour_threshold] = 0
        contour_indices = np.nonzero(contour)
        contour_indices = np.stack(contour_indices, axis=1)

        # 进行曲线简化
        if self.use_naive_simplify:
            current_length = contour_indices.shape[0]
            if current_length > self.contour_max_points:
                simplified_indices = (np.arange(self.contour_max_points, dtype=np.float32) * current_length / self.contour_max_points).astype(np.int32)
                contour_indices = contour_indices[simplified_indices, :]
        else:
            while contour_indices.shape[0] > self.contour_max_points:
                last_length = contour_indices.shape[0]
                epsilon = self.coutour_simplify_eplision * cv2.arcLength(contour_indices, closed=True)
                contour_indices = cv2.approxPolyDP(contour_indices, epsilon=epsilon, closed=True)
                if contour_indices.shape[0] == last_length:
                    break
        
        if len(contour_indices.shape) == 3:
            contour_indices = contour_indices[:, 0, :]
        return torch.from_numpy(contour_indices).to(device) if is_tensor else contour_indices