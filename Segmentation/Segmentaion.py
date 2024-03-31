import os
from tqdm import tqdm
import torch
from Dataset.RHD import RHDDataset, RHDDatasetItem
from Config import Config
import pickle

def segment_image(image, config):
    pass

class SkinColorSegmentation(object):
    """
    肤色分割方法
    """
    def __init__(self, dataset: RHDDataset, config: Config):
        self.dataset = dataset
        self.config = config
        self.device = dataset.device
        self.model_path = f"{os.path.dirname(os.path.abspath(__file__))}/model/{self.config.dataset_name}.pickle"

    def load_trained_model(self, max_size=-1, force_retrain=False) -> torch.Tensor:
        """
        加载训练好的模型
        :param max_size: 最大训练集大小，-1表示使用全部数据
        :param force_retrain: 强制重新训练模型
        """
        if not self.__have_trained_model() or force_retrain:
            model = self.train_model(max_size)
            if not os.path.exists(os.path.dirname(self.model_path)):
                os.makedirs(os.path.dirname(self.model_path))
            torch.save(model, self.model_path)
        else:
            model = torch.load(self.model_path)
        return model

    def train_model(self, max_size=-1) -> torch.Tensor:
        """
        训练模型
        :param max_size: 最大训练集大小，-1表示使用全部数据
        """
        sample_count = len(self.dataset) if max_size == -1 or max_size > len(self.dataset) else max_size
        image_size = self.dataset.image_shape[0] * self.dataset.image_shape[1]
        ps = 0      # 皮肤色的频率
        pc: torch.Tensor = torch.zeros((65536,), dtype=torch.int32, device=self.device)      # 各种颜色出现的频率 按UV空间分割 (65536)
        pcs: torch.Tensor = torch.zeros((65536,), dtype=torch.int32, device=self.device)     # 各种颜色在皮肤色中出现的频率 按UV空间分割 (65536)
        
        print("开始训练模型...")
        for i in tqdm(range(sample_count)):
            sample = self.dataset[i]
            u = sample.color[:, :, 1].flatten().int()
            v = sample.color[:, :, 2].flatten().int()
            uv = u * 256 + v
            hand_mask = sample.mask.flatten() >= 2      # 手掌掩膜 分类>=2为手掌的部分

            ps_cur = hand_mask.sum(where=hand_mask)
            
            pc_cur = torch.bincount(uv, minlength=pc.numel())
            pcs_cur = torch.bincount(uv[hand_mask], minlength=pcs.numel())
            ps += ps_cur
            pc += pc_cur
            pcs += pcs_cur

        # 频率转换为概率
        ps = ps / (sample_count * image_size)
        pc = pc.float() / (sample_count * image_size)
        pcs = pcs.float() / (ps * sample_count * image_size)
        print(f"训练完成 皮肤色概率: {ps:.6f}")
        print(f"各种颜色出现的概率之和: {float(pc.sum()):.6f}")
        print(f"各种颜色在皮肤色中出现的概率之和: {float(pcs.sum()):.6f}")

        # 计算一个颜色是皮肤色的概率P(S|C)
        result: torch.Tensor = torch.zeros((65536,), dtype=torch.float32, device=self.device)
        valid_mask = pc > 0
        result[valid_mask] = pcs[valid_mask] * ps / pc[valid_mask]
        return result.view(256, 256)

    def __have_trained_model(self) -> bool:
        return os.path.exists(self.model_path)