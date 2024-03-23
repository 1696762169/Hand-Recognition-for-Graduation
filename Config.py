import os
import yaml
from Dataset.RHD import RHDDataset

class Config(object):
    def __init__(self, config_path: str):
        if not config_path.endswith('.yaml'):
            config_path += '.yaml'
        with open(os.path.join(os.getcwd(), 'Config', config_path), 'r', encoding='utf-8') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        # 数据集参数
        dataset = self.config['dataset']
        self.dataset_name = dataset['type']
        self.dataset_name = dataset['name']
        self.dataset_root = dataset['root']
        self.dataset_split = dataset['split_type']
        self._dataset = self.__create_dataset()
        
    def __create_dataset(self):
        params = {}
        if 'params' in self.config['dataset']:
            params = self.config['dataset']['params']
            params = params if isinstance(params, dict) else {}
        
        if self.dataset_name == 'RHD':
            return RHDDataset(self.dataset_root, self.dataset_split, **params)
        else:
            raise ValueError('Unsupported dataset type: {}'.format(self.dataset_name))

    @property
    def dataset(self):
        return self._dataset