import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier


from Dataset.RHD import RHDDataset, RHDDatasetItem
from Config import Config
from matplotlib import pyplot as plt
from typing import List, Tuple

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
        self.model = None

    def segment(self, image: RHDDatasetItem) -> List[Tuple[torch.Tensor, int]]:
        """
        对单张图片进行肤色分割
        :param image: 单张图片
        :return: 肤色分割结果 [(区域掩膜, 区域大小),...]
        """
        if self.model is None:
            self.load_trained_model()
        
        color = image.color
        rgb = image.color_rgb
        seed_mask = self.__get_uv_mask(color, self.model, self.config.seg_seed_threshold)
        regioin_mask = self.__get_uv_mask(color, self.model, self.config.seg_blob_threshold)
        seed_indices = seed_mask.nonzero(as_tuple=False)

        # 遍历所有种子点 进行区域生长
        regions = []
        for i in range(seed_indices.shape[0]):
            seed = seed_indices[i]
            region = self.__grow_region(seed, regioin_mask)
            # 去除已经被包含的区域
            seed_mask[region] = False
            regioin_mask[region] = False
            # 去除过小的区域
            region_size = region.count_nonzero()
            if region_size < self.config.seg_min_region_size:
                continue
            # 利用卷积扩展区域
            extend_radius = 5
            kernal = torch.ones((1, 1, extend_radius * 2 + 1, extend_radius * 2 + 1), dtype=torch.float32, device=self.device, requires_grad=False)
            region = F.conv2d(region.float()[None, None, :, :], kernal, padding=extend_radius).squeeze() > 0
            regions.append((region, region_size))

        # 按区域大小排序
        regions.sort(key=lambda x: x[1], reverse=True)

        region_count = len(regions)
        plt.subplot(1, region_count + 1, 1)
        plt.imshow(rgb)
        for i in range(region_count):
            plt.subplot(1, region_count + 1, i + 2)
            plt.imshow(regions[i][0].cpu().numpy())

        plt.show()
        return regions

    def __get_uv_mask(self, yuv: torch.Tensor, uv_model: torch.Tensor, threshold: float) -> torch.Tensor:
        u = yuv[:, :, 1].flatten().int()
        v = yuv[:, :, 2].flatten().int()
        value = uv_model[u, v]
        mask = value > threshold
        return mask.view(yuv.shape[0], yuv.shape[1])
    
    def __grow_region(self, seed: torch.Tensor, region_mask: torch.Tensor) -> torch.Tensor:
        """
        区域生长算法
        :param seed: 种子点 (2)
        :param mask: 可生长区域的掩膜 (w*h)
        :return: 生长后的区域 (w*h)
        """
        region = torch.zeros_like(region_mask, dtype=torch.bool, requires_grad=False)
        region[seed[0], seed[1]] = 1.0
        pre_region = torch.zeros_like(region_mask, dtype=torch.bool, requires_grad=False)
        kernal = torch.tensor([[1, 1, 1], 
                               [1, 0, 1], 
                               [1, 1, 1]], dtype=torch.float32, device=self.device, requires_grad=False)
        kernal = kernal[None, None, :, :]
        
        # count = 0
        while not torch.equal(region, pre_region):
            pre_region[:, :] = region
            # 利用卷积扩展区域
            region = F.conv2d(region.float()[None, None, :, :], kernal, padding=1).squeeze() > 0
            region &= region_mask
            # count += 1
            # if count % 10 == 0:
            #     plt.imshow(region.float().cpu().numpy())
                # plt.show()
        return region

    def load_trained_model(self, max_size=-1, force_retrain=False) -> torch.Tensor:
        """
        加载训练好的模型
        :param max_size: 最大训练集大小，-1表示使用全部数据
        :param force_retrain: 强制重新训练模型
        """
        if not self.__have_trained_model() or force_retrain:
            self.model = self.train_model(max_size)
            if not os.path.exists(os.path.dirname(self.model_path)):
                os.makedirs(os.path.dirname(self.model_path))
            torch.save(self.model, self.model_path)
        else:
            self.model = torch.load(self.model_path)
        return self.model

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
        with torch.no_grad():
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
    
class RDFSegmentor(object):
    """
    随机森林分割方法
    """
    def __init__(self, config: Config,
                 *,
                 n_estimators=3,    # 随机森林的树的数量
                 max_depth=None,    # 树的最大深度
                 min_samples_split=2,  # 节点分裂所需的最小样本数
                 min_samples_leaf=1,   # 叶子节点最少包含的样本数
                 random_state=None,   # 随机数种子
                 ):
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion='entropy', 
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            )
        self.min_samples_split = min_samples_split
        self.config = config