import os
import PIL.Image as Image
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Literal

class RHDDatasetItem(object):
    def __init__(self, img_index: int, anno: dict, data_root: str, device: torch.device) -> None:
        self.img_index = img_index
        self.img_name = str.format("{0:05d}.png", self.img_index)
        self.data_root = data_root

        self.kp_coord_uv = anno['uv_vis'][:, :2]  # 42 个关键点的 2D UV 坐标，单位为像素 (42, 2)
        self.kp_visible = (anno['uv_vis'][:, 2] == 1)  # 42 个关键点的可见性标志，1 为可见，0 为不可见
        self.kp_coord_xyz = anno['xyz']  # 42 个关键点的 3D XYZ 坐标，单位为 mm (42, 3)
        self.camera_intrinsic_matrix = anno['K']  # 相机内参矩阵 (3, 3)

        # RGB 图像
        self.color = Image.open(os.path.join(self.data_root, 'color', self.img_name))
        self.color = np.array(self.color, dtype=np.float32) / 255.0   # 图像归一化到 [0, 1]
        # 深度图像
        self.depth = Image.open(os.path.join(self.data_root, 'depth', self.img_name))
        self.depth = np.array(self.depth, dtype=np.int32)
        self.depth = self.depth[:, :, 0] * 256 + self.depth[:, :, 1]  # 深度值范围 [0, 65535]
        # 分类掩码
        self.mask = Image.open(os.path.join(self.data_root,'mask', self.img_name))
        self.mask = np.array(self.mask, dtype=np.int8)

        if device is not None:
            self.color = self.__to_device(self.color, device)
            self.depth = self.__to_device(self.depth, device)
        pass

    def __to_device(self, array: np.ndarray, device: torch.device) -> torch.Tensor:
        return torch.from_numpy(array).to(device)


class RHDDataset(Dataset):
    def __init__(self,
                 data_root: str,
                 set_type: Literal["training", "evaluation"] = 'training',
                 *,
                 to_tensor: bool = True,
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 shared_list: list = None) -> None:
        self.set_type = set_type
        self.to_tensor = to_tensor
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
            
    def __load_annotations(self):
        with open(os.path.join(self.data_root, f"anno_{self.set_type}.pickle"), 'rb') as f:
            annotations = pickle.load(f)
        return annotations
    
    def __load_item(self, index) -> RHDDatasetItem:
        return RHDDatasetItem(index, self.annotations[index], self.data_root, self.device if self.to_tensor else None)
