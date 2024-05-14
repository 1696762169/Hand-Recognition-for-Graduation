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
        if 'dataset' in self.config:
            dataset = self.config['dataset']
            self.dataset_type = dataset['type']
            self.dataset_name = dataset['name']
            self.dataset_root = dataset['root']
            self.dataset_split = dataset['split_type']

        # 分割算法参数
        if'segmentation' in self.config:
            segmentation = self.config['segmentation']
            self.seg_type = segmentation['type']

        # 评价参数
        if 'evaluation' in self.config:
            evaluation = self.config['evaluation']
            self.depth_max_diff = evaluation['depth_max_diff']
            self.depth_match_threshold = evaluation['depth_match_threshold']
       
    def get_dataset_params(self) -> dict:
        if 'params' in self.config['dataset']:
            params = self.config['dataset']['params']
            params = params if isinstance(params, dict) else {}
            return params
        else:
            return {}

    def get_seg_params(self) -> dict:
        if 'params' in self.config['segmentation']:
            params = self.config['segmentation']['params']
            params = params if isinstance(params, dict) else {}
            return params
        else:
            return {}