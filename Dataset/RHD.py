import os
import PIL.Image as Image
import cv2
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Literal

class RHDDatasetItem(object):
    def __init__(self, img_index: int, anno: dict, data_root: str, to_yuv: bool, device: torch.device, use_modified_depth: bool = False) -> None:
        self.img_index = img_index
        self.img_name = str.format("{0:05d}.png", self.img_index)
        self.data_root = data_root
        self.to_yuv = to_yuv
        self.device = device

        self.kp_coord_uv = anno['uv_vis'][:, :2]  # 42 个关键点的 2D UV 坐标，单位为像素 (42, 2)
        self.kp_visible = (anno['uv_vis'][:, 2] == 1)  # 42 个关键点的可见性标志，1 为可见，0 为不可见
        self.kp_coord_xyz = anno['xyz']  # 42 个关键点的 3D XYZ 坐标，单位为 mm (42, 3)
        self.camera_intrinsic_matrix = anno['K']  # 相机内参矩阵 (3, 3)

        # 彩色 图像
        self.color = cv2.imread(os.path.join(self.data_root, 'color', self.img_name), cv2.IMREAD_UNCHANGED)
        self.color = cv2.cvtColor(self.color, cv2.COLOR_BGR2YUV if self.to_yuv else cv2.COLOR_BGR2RGB)
        # self.color = self.color.astype(np.float32) / 256.0  # 图像归一化到 [0, 1]
        
        # 深度图像
        if use_modified_depth:
            self.depth = cv2.imread(os.path.join(self.data_root, 'modified_depth', self.img_name), cv2.IMREAD_GRAYSCALE)
            self.depth = self.depth / 255.0  # 深度值归一化到 [0, 1]
        else:
            self.depth = cv2.imread(os.path.join(self.data_root, 'depth', self.img_name), cv2.IMREAD_UNCHANGED)
            self.depth = (self.depth[:, :, 2] * 256 + self.depth[:, :, 1]) / 65536.0  # 深度值归一化到 [0, 1]
        self.depth = self.depth.astype(np.float32)

        # 分类掩码
        self.origin_mask = cv2.imread(os.path.join(self.data_root,'mask', self.img_name), cv2.IMREAD_UNCHANGED).astype(np.int8)
        self.mask = np.zeros_like(self.origin_mask)
        self.mask[self.origin_mask >= 2] = 1  # 手部
        # self.mask = self.origin_mask.copy()
        # self.mask[(self.origin_mask >= 2) & (self.origin_mask <= 17)] = 2  # 左手
        # self.mask[(self.origin_mask >= 18) & (self.origin_mask <= 34)] = 3  # 右手

        if device is not None:
            self.color = self.__to_device(self.color, device)
            self.depth = self.__to_device(self.depth, device)
        pass

    def __to_device(self, array: np.ndarray, device: torch.device) -> torch.Tensor:
        return torch.from_numpy(array).to(device)
    
    @property
    def color_rgb(self) -> np.ndarray:
        if self.device is None:
            return cv2.cvtColor(self.color, cv2.COLOR_YUV2RGB) if self.to_yuv else self.color
        else:
            return cv2.cvtColor(self.color.cpu().numpy(), cv2.COLOR_YUV2RGB) if self.to_yuv else self.color.cpu().numpy()
    
    @property
    def color_path(self) -> str:
        return os.path.join('.\\color', self.img_name)


class RHDDataset(Dataset):
    """
    RHD 数据集
    Keypoints available:
    0: left wrist, 1-4: left thumb [tip to palm], 5-8: left index, ..., 17-20: left pinky,
    21: right wrist, 22-25: right thumb, ..., 38-41: right pinky

    Segmentation masks available:
    0: background, 1: person, 
    2-4: left thumb [tip to palm], 5-7: left index, ..., 14-16: left pinky, 17: palm, 
    18-20: right thumb, ..., right palm: 33
    """
    def __init__(self,
                 data_root: str,
                 set_type: Literal["train", "test"] = 'train',
                 to_tensor: bool = True,
                 *,
                 use_modified_depth: bool = False,
                 to_yuv: bool = False,
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 shared_list: list = None) -> None:
        self.set_type = set_type
        self.to_tensor = to_tensor
        self.use_modified_depth = use_modified_depth
        self.to_yuv = to_yuv
        self.device = device
        self.data_root = os.path.join(data_root, set_type)
        self.annotations = self.__load_annotations()

        self.shared_list = shared_list
        if self.shared_list is not None:
            self.shared_list = [None for _ in range(len(self.annotations))]

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        if self.shared_list is not None:
            if self.shared_list[index] is not None:
                return self.shared_list[index]
            self.shared_list[index] = self.__load_item(index)
            return self.shared_list[index]
        else:
            return self.__load_item(index)
    
    @property
    def image_shape(self) -> tuple:
        return self[0].color.shape[:2]
    
    @property
    def pixel_count(self) -> int:
        shape = self.image_shape
        return shape[0] * shape[1]
            
    def __load_annotations(self):
        with open(os.path.join(self.data_root, f"anno_{self.set_type}.pickle"), 'rb') as f:
            annotations = pickle.load(f)
        return annotations
    
    def __load_item(self, index) -> RHDDatasetItem:
        return RHDDatasetItem(index, 
                              self.annotations[index], 
                              self.data_root, 
                              self.to_yuv, 
                              self.device if self.to_tensor else None, 
                              self.use_modified_depth)

    def is_single_hand(self, index: int) -> bool:
        """
        判断指定索引的样本是否只包含一个手部区域
        """
        sample = self[index]
        mask = sample.mask
        n_label, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        return n_label == 2
