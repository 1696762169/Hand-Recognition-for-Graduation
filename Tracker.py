import cv2
import torch
from scipy import ndimage
from torch.nn import functional as F
from torchvision import transforms
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

                 length_block=5,    # 弯曲块长度区间数量
                 angle_block=12,    # 弯曲块角度区间数量
                 ):
        self.device = device

        # 轮廓简化参数
        self.contour_threshold = contour_threshold
        self.contour_max_points = contour_max_points
        self.coutour_simplify_eplision = coutour_simplify_eplision
        self.use_naive_simplify = use_naive_simplify

        # 弯曲块参数
        self.length_block = length_block
        self.angle_block = angle_block

    def __call__(self, depth_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        提取3D-SC描述符特征
        :param depth_map: 深度图 (B, H, W) [0, 1]
        :param mask: 预测手部掩码图 (B, H, W) 0/1
        :return: 3D-SC描述符特征 (point_count, length_block, angle_block)
        """
        # 提取轮廓
        contour = self.extract_contour(depth_map, mask)
        # 简化轮廓
        simplified_contour = self.simplify_contour(contour)
        # 计算特征
        features = self.get_descriptor_features(simplified_contour, depth_map)
        return features

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
        final_threshold = torch.mean(depth_map[depth_mask]) + 0.1
        grad_magnitude[:, 0, :, :][depth_map > final_threshold] = 0     # 最后过滤掉可能被计入的背景值

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
    
    def get_descriptor_features(self, contour: torch.Tensor, depth_map: torch.Tensor) -> torch.Tensor:
        """
        获取3D-SC描述符特征
        :param contour: 轮廓特征 (point_count, 2)
        :param depth_map: 深度图 (H, W) [0, 1]
        :return: 3D-SC描述符特征 (point_count, length_block, angle_block)
        """
        point_count = contour.shape[0]
        contour = contour.int().to(self.device)
        depth_map = depth_map.float().to(self.device)

        # 计算各个边缘点之间的距离向量 (dy, dx, dz)
        depth = depth_map[contour[:, 0], contour[:, 1]]
        relative_depth = torch.repeat_interleave((depth - depth.min()).unsqueeze(1), point_count, dim=1)
        diff_vec = (contour[:, None, :] - contour[None, :, :]).permute(2, 0, 1).float()
        diff_vec += torch.eye(point_count, device=self.device)[None, :, :] * 1e-6

        # 极坐标向量
        angle = torch.atan2(diff_vec[0], diff_vec[1])
        log_len = torch.log(torch.norm(diff_vec, p=2, dim=0))
        len_max = log_len.max().item()

        # 计算弯曲块 将数值离散化为特征区间
        angle_bins = torch.linspace(-torch.pi - 1e-4, torch.pi, self.angle_block + 1, device=self.device)
        length_bins = torch.linspace(- 1e-4, len_max, self.length_block + 1, device=self.device)
        angle_indices = torch.bucketize(angle, angle_bins) - 1
        length_indices = torch.bucketize(log_len, length_bins) - 1
        
        # 计算3D-SC描述符特征
        features = torch.zeros((self.length_block, self.angle_block, point_count), device=self.device)
        for l in range(self.length_block):
            for a in range(self.angle_block):
                mask = ((length_indices == l) & (angle_indices == a)).float()
                # test = relative_depth * mask
                features[l, a] = torch.sum(relative_depth * mask, dim=1)
        
        return features.permute(2, 0, 1)


class LBPTracker(object):
    """
    LBP特征生产器
    """
    def __init__(self,
                 *,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 radius: float=2.0,
                 n_points: int=8,):
        
        self.device = device

        self.radius = radius
        self.n_points = n_points

        # 缓存采样偏移向量
        angles = np.arange(0, 2*np.pi, 2*np.pi/self.n_points)
        self.offsets = np.stack((np.sin(angles), np.cos(angles)), axis=0) * self.radius
        self.offsets = torch.from_numpy(self.offsets).float().to(self.device)

    def __call__(self, depth_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        提取LBP描述符特征
        :param depth_map: 深度图 (H, W) [0, 1]
        :param mask: 预测手部掩码图 (H, W) 0/1
        :return: LBP描述符特征 (H, W)
        """
        mask = transforms.Resize(depth_map.shape, interpolation=transforms.InterpolationMode.NEAREST)(mask.unsqueeze(0))[0]

        # 获取采样网格
        indices = torch.nonzero(mask, as_tuple=False).float().to(self.device)
        if indices.shape[0] == 0:
            return torch.zeros(depth_map.shape[0], device=self.device)
        indices = torch.repeat_interleave(indices[:, :, None], self.n_points, dim=2) # (n_mask, 2, n_points)
        ori_indices = indices.clone().permute(2, 0, 1).int()  # (n_points, n_mask, 2)
        indices += self.offsets[None, :, :]
        indices = indices.permute(2, 0, 1)[:, None, :, :] # (n_points, 1, n_mask, 2)

        # 采样深度
        stack_depths = torch.repeat_interleave(depth_map[None, :, :], self.n_points, dim=0) # (n_points, H, W)
        shape = torch.tensor(depth_map.shape, device=self.device)[None, None, None, :]
        norm_indices = indices / (shape * 0.5) - 1  # 坐标归一化 [-1, 1]
        norm_indices = norm_indices.flip(3)  # 坐标反转 每对坐标 (y, x) -> (x, y)
        sample_depths = F.grid_sample(stack_depths[:, None, :, :], norm_indices, align_corners=True)[:, 0, 0, :]    # (n_points, n_mask)
        
        # 还原采样的深度图形状
        indices = indices[:, 0, :, :].int()  # (n_points, n_mask, 2)
        valid = (indices[:, :, 0] >= 0) & (indices[:, :, 0] < depth_map.shape[0]) & (indices[:, :, 1] >= 0) & (indices[:, :, 1] < depth_map.shape[1])
        valid_indices = ori_indices[valid, :]   # (n_valid, 2)
        point_indices = torch.nonzero(valid)[:, 0]
        offset_depths = torch.zeros((self.n_points, depth_map.shape[0], depth_map.shape[1]), device=self.device)  # (n_points, H, W)
        offset_depths[point_indices, valid_indices[:, 0], valid_indices[:, 1]] = sample_depths[valid]

        # 计算LBP特征
        lbp = (offset_depths >= depth_map.unsqueeze(0)).byte()   # (n_points, H, W)
        lbp = lbp * torch.pow(2, torch.arange(self.n_points, device=self.device))[:, None, None]
        lbp = lbp.sum(dim=0)
        return lbp
        
        col = 4
        plt.subplot(1, col, 1)
        plt.imshow(depth_map.cpu().numpy(), cmap='gray')
        plt.subplot(1, col, 2)
        plt.imshow(mask.cpu().numpy(), cmap='gray')
        plt.subplot(1, col, 3)
        # offset_depths = sample_depths.cpu().numpy()
        # offset_depths = np.zeros(depth_map.shape)
        # cpu_indices = indices[0, :, :].cpu().numpy()
        # cpu_valid = valid[0, :].cpu().numpy()
        # cpu_indices = cpu_indices[cpu_valid, :]
        # offset_depths[cpu_indices[:, 0], cpu_indices[:, 1]] = sample_depths[0].cpu().numpy()[cpu_valid]
        offset_depths = offset_depths.cpu().numpy()
        plt.imshow(offset_depths[5], cmap='gray')
        plt.subplot(1, col, 4)
        plt.imshow(lbp.cpu().numpy(), cmap='gray')
        plt.show()
        return lbp
