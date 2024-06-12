import os
import cv2
import pickle
import struct
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from scipy.ndimage import distance_transform_edt
from typing import Literal
import logging

class SenzDatasetItem(object):
    def __init__(self, 
                 data_root: str, 
                 subject_index: int, 
                 ges_index: int, 
                 img_index: int, 
                 to_yuv: bool, 
                 filter_type: Literal['none', 'gaussian', 'median', 'bilateral'], 
                 device: torch.device) -> None:
        self.subject = subject_index
        self.label = ges_index
        self.rgb_path = f'./S{subject_index}/G{ges_index}/{img_index}-color.png'
        path_prefix = f'{data_root}/S{subject_index}/G{ges_index}/{img_index}'
        self.to_yuv = to_yuv
        self.device = device

        # 彩色图像
        self.color = cv2.imread(path_prefix + '-color.png', cv2.IMREAD_UNCHANGED)
        self.color = cv2.cvtColor(self.color, cv2.COLOR_BGR2YUV if self.to_yuv else cv2.COLOR_BGR2RGB)
        # self.color = self.color.astype(np.float32) / 256.0  # 图像归一化到 [0, 1]
        
        # 深度图像
        dw, dh = 320, 240
        self.depth = np.fromfile(path_prefix + '-depth.bin', dtype=np.int16, count=dw*dh)
        self.depth = self.depth.reshape((dh, dw)).astype(np.float32) / 65536.0  # 图像归一化到 [0, 1]
        self.depth = cv2.GaussianBlur(self.depth, (5, 5), 1)
        self.depth = cv2.medianBlur(self.depth, 5)
        if filter_type == 'gaussian':
            self.depth = cv2.GaussianBlur(self.depth, (5, 5), 1)
        elif filter_type =='median':
            self.depth = cv2.medianBlur(self.depth, 5)
        elif filter_type == 'bilateral':
            self.depth = cv2.bilateralFilter(self.depth, d=9, sigmaSpace=75, sigmaColor=0.1)

        # 置信度掩码
        self.confidence = np.fromfile(path_prefix + '-conf.bin', dtype=np.int16, count=dw*dh).reshape((dh, dw))
        threshold, mask = cv2.threshold(self.confidence.astype(np.uint8), 0, 65535, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        self.confidence = self.confidence.astype(np.float32) / 65536.0  # 图像归一化到 [0, 1]
        # print(f'阈值：{threshold}')
        self.mask = self.confidence > threshold / 65536.0 * 5 # 掩码
        # self.mask = mask != 0  # 掩码

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
        return self.rgb_path



class SenzDataset(Dataset):
    """
    Kinect Leap 数据集 （继承自Senz3d）
    r-depth.bin: Senz3D depth map raw (320 x 240 short 16 bit).
    r-conf.bin: Senz3D confidence map raw (320 x 240 short 16 bit).
    r-color.png: Senz3D color map (640 x 480).
    where 1<r=<=30 is the number of repetition.
    """
    def __init__(self,
                 data_root: str,
                 set_type: Literal["train", "test"] = 'train',
                 to_tensor: bool = True,
                 *,
                 to_yuv: bool = False,
                 filter_type: Literal['none', 'gaussian', 'median', 'bilateral'] = 'none',

                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 buffer_size: int = 0,
                 shared_list: dict[int, SenzDatasetItem] = None) -> None:
        super().__init__()
        self.data_root = os.path.join(data_root, 'data')
        self.set_type = set_type
        self.to_tensor = to_tensor
        self.to_yuv = to_yuv
        self.filter_type = filter_type
        self.device = device
        self.buffer_size = buffer_size

        # 读取文件列表
        self.meta_data : list[tuple[int, int, int]] = []  # (subject_index, ges_index, img_index)
        with open(os.path.join(data_root, 'data.txt'), 'r') as f:
            for line in f.readlines():
                _, sub_idx, ges_idx, img_idx = line.strip().split(' ')
                self.meta_data.append((int(sub_idx), int(ges_idx), int(img_idx)))
        self.labels = [ges_idx for _, ges_idx, _ in self.meta_data]

        self.shared_list = shared_list or {}
        self.buffer_queue = []

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, index):
        if self.buffer_size == 0:
            return self.__load_item(index)
        if index in self.shared_list:
            return self.shared_list[index]
        if len(self.buffer_queue) >= self.buffer_size:
            self.shared_list.pop(self.buffer_queue.pop(0))
        item = self.__load_item(index)
        self.buffer_queue.append(index)
        self.shared_list[index] = item
        return self.shared_list[index]

    def __load_item(self, index) -> SenzDatasetItem:
        meta_data = self.meta_data[index]
        logging.debug(f'正在加载Senz3D数据集：{index}')
        device = self.device if self.to_tensor else None
        item = SenzDatasetItem(self.data_root, meta_data[0], meta_data[1], meta_data[2], self.to_yuv, self.filter_type, device)
        return item