import os
import yaml
from Dataset.RHD import RHDDataset
from Dataset.IsoGD import IsoGDDataset

class Config(object):
    def __init__(self, 
                 *,
                 dataset_type: str | None = None,
                 seg_type: str | None = None,
                 config_path: str | None = None):
        
        # 加载配置文件
        config_dir = os.path.join(os.getcwd(), 'Config')
        if config_path is not None:
            if not config_path.endswith('.yaml'):
                config_path += '.yaml'
            with open(os.path.join(config_dir, config_path), 'r', encoding='utf-8') as f:
                self.config = yaml.load(f, Loader=yaml.FullLoader)
        else:
            self.config = {}

        # 数据集参数
        if 'dataset_type' in self.config:
            dataset_type = self.config['dataset_type']
        if 'dataset' not in self.config and dataset_type is not None:
            with open(os.path.join(config_dir, f'data_{dataset_type}.yaml'), 'r', encoding='utf-8') as f:
                self.config['dataset'] = yaml.load(f, Loader=yaml.FullLoader)['dataset']
        else:
            self.config['dataset'] = {}
        if len(self.config['dataset']) > 0:
            dataset = self.config['dataset']
            self.dataset_type = dataset['type']
            self.dataset_name = dataset['name']
            self.dataset_root = dataset['root']
            self.dataset_split = dataset['split_type']
            self.dataset_to_tensor = dataset['to_tensor']

        # 分割算法参数
        if 'seg_type' in self.config:
            seg_type = self.config['seg_type']
        if 'segmentation' not in self.config and seg_type is not None:
            with open(os.path.join(config_dir, f'seg_{seg_type}.yaml'), 'r', encoding='utf-8') as f:
                self.config['segmentation'] = yaml.load(f, Loader=yaml.FullLoader)['segmentation']
        else:
            self.config['segmentation'] = {}
        if len(self.config['segmentation']) > 0:
            segmentation = self.config['segmentation']
            self.seg_type = segmentation['type']
            self.seg_model_path = segmentation['model_path']

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