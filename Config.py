import os
import yaml
from Dataset.RHD import RHDDataset
from Dataset.IsoGD import IsoGDDataset

class Config(object):
    def __init__(self, config_path: str):
        if not config_path.endswith('.yaml'):
            config_path += '.yaml'
        with open(os.path.join(os.getcwd(), 'Config', config_path), 'r', encoding='utf-8') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        # 数据集参数
        dataset = self.config['dataset']
        self.dataset_type = dataset['type']
        self.dataset_name = dataset['name']
        self.dataset_root = dataset['root']
        self.dataset_split = dataset['split_type']
        self._dataset = self.__create_dataset()

        # 分割算法参数
        segmentation = self.config['segmentation']
        self.seg_seed_threshold = segmentation['seed_threshold']
        self.seg_blob_threshold = segmentation['blob_threshold']
        self.seg_min_region_size = segmentation['min_region_size']

        # 评价参数
        evaluation = self.config['evaluation']
        self.depth_max_diff = evaluation['depth_max_diff']
        self.depth_match_threshold = evaluation['depth_match_threshold']
        
    def __create_dataset(self):
        params = {}
        if 'params' in self.config['dataset']:
            params = self.config['dataset']['params']
            params = params if isinstance(params, dict) else {}
        
        if self.dataset_type == 'RHD':
            return RHDDataset(self.dataset_root, self.dataset_split, **params)
        elif self.dataset_type == 'IsoGD':
            return IsoGDDataset(self.dataset_root, self.dataset_split, **params)
        else:
            raise ValueError('Unsupported dataset type: {}'.format(self.dataset_type))

    @property
    def dataset(self):
        return self._dataset