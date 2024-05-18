import torch
from torch.nn import functional as F
import numpy as np

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

    def extract_contour(self, depth_map: torch.Tensor) -> torch.Tensor:
        """
        提取轮廓特征
        :param depth_map: 深度图 (B, H, W)
        :return: 轮廓特征 (B, H, W)
        """
        origin_shape = depth_map.shape
        single_input = len(origin_shape) == 2
        depth_map = depth_map[None, None, :, :] if single_input else depth_map[:, None, :, :]

        # 计算梯度幅度
        sobel_op = SobelOperator(self.device)
        grad_magnitude: torch.Tensor = sobel_op(depth_map.float().to(self.device))
        grad_magnitude = grad_magnitude.detach()

        return grad_magnitude[0, 0, :, :] if single_input else grad_magnitude[:, 0, :, :]