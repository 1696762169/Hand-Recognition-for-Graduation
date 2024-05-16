import os
import cv2
import torch
import logging
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from typing import Literal

class IsoGDDatasetItem(object):
    def __init__(self, rgb_path: str, depth_path: str, label: int, rgb_transform=None, depth_transform=None):
        self.rgb_path = rgb_path
        self.depth_path = depth_path
        self.label = label
        rgb_transform = rgb_transform or (lambda x: np.array(x))
        depth_transform = depth_transform or (lambda x: np.array(x))

        # 读取RGB视频
        self.rgb_frames: list[torch.Tensor] = []
        cap = cv2.VideoCapture(self.rgb_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            self.rgb_frames.append(rgb_transform(frame))
        cap.release()

        # 读取深度视频
        self.depth_frames: list[torch.Tensor] = []
        cap = cv2.VideoCapture(self.depth_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            self.depth_frames.append(depth_transform(frame))
        cap.release()

        assert len(self.rgb_frames) == len(self.depth_frames)

    def __len__(self):
        return len(self.rgb_frames)
    def __getitem__(self, index):
        return (self.rgb_frames[index], self.depth_frames[index], self.label)
    
    def get_rgb_frame(self, index) -> torch.Tensor:
        return self.rgb_frames[index]
    def get_depth_frame(self, index) -> torch.Tensor:
        return self.depth_frames[index]

class IsoGDDataset(Dataset):
    def __init__(self, 
                 data_root: str, 
                 set_type: Literal['train', 'valid', 'test'],
                 to_tensor: bool = True,
                 *,
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 buffer_size: int = 0,
                 shared_list: dict[int, IsoGDDatasetItem] = None) -> None:
        super().__init__()
        self.data_root = data_root
        self.set_type = set_type
        self.to_tensor = to_tensor
        self.device = device
        self.buffer_size = buffer_size

        # 读取文件列表
        self.meta_data : list[tuple[str, str, int]] = []  # (rgb_path, depth_path, label)
        with open(os.path.join(data_root, set_type + '.txt'), 'r') as f:
            for line in f.readlines():
                rgb_path, depth_path, label = line.strip().split(' ')
                self.meta_data.append((rgb_path, depth_path, int(label)))

        # 定义transform
        def default_depth_transform(x):
            x: np.ndarray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            x = x.astype(np.float32) / 255.0
            return torch.from_numpy(x).float()
        
        self.rgb_transform = transforms.ToTensor() if to_tensor else None
        self.depth_transform = default_depth_transform if to_tensor else None

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

    def __load_item(self, index) -> IsoGDDatasetItem:
        meta_data = self.meta_data[index]
        logging.debug(f'正在加载数据集：{index}')
        return IsoGDDatasetItem(os.path.join(self.data_root, meta_data[0]), 
                                os.path.join(self.data_root, meta_data[1]), 
                                meta_data[2], 
                                self.rgb_transform,
                                self.depth_transform)
    
    def collate_fn(self, batch):
        return [item for item in batch]