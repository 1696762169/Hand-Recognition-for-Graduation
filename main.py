               #########                       
              ############                     
              #############                    
             ##  ###########                   
            ###  ###### #####                  
            ### #######   ####                 
           ###  ########## ####                
          ####  ########### ####               
         ####   ###########  #####             
        #####   ### ########   #####           
       #####   ###   ########   ######         
      ######   ###  ###########   ######       
     ######   #### ##############  ######      
    #######  #####################  ######     
    #######  ######################  ######    
   #######  ###### #################  ######   
   #######  ###### ###### #########   ######   
   #######    ##  ######   ######     ######   
   #######        ######    #####     #####    
    ######        #####     #####     ####     
     #####        ####      #####     ###      
      #####       ###        ###      #        
        ###       ###        ###               
         ##       ###        ###               
#_________#_______####_______####______________#
             #我们的未来没有BUG#
# ref: https://github.com/leinlin/Miku-LuaProfiler/

import os
import sys
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
import argparse
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
import time

from Config import Config
from PSO import PSOSolver
from Utils import Utils
from HandModel import HandModel
from Evaluation import Evaluation
from Segmentation import ResNetSegmentor, RDFSegmentor
from Dataset.RHD import RHDDataset
from Dataset.IsoGD import IsoGDDataset
from Dataset.Senz import SenzDataset
import Test

def test_mask(dataset: RHDDataset):
    sample = dataset[0]
    rgb = cv2.cvtColor(sample.color, cv2.COLOR_YUV2RGB)
    person = np.zeros_like(sample.origin_mask)
    person[sample.origin_mask == 1] = 1
    hand = np.zeros_like(sample.origin_mask)
    hand[sample.origin_mask >= 2] = 1

    plt.figure()

    plt.subplot(1, 3, 1)
    plt.imshow(rgb)
    plt.axis('off')
    plt.title('RGB')

    plt.subplot(1, 3, 2)
    plt.imshow(person)
    plt.axis('off')
    plt.title('Person')

    plt.subplot(1, 3, 3)
    plt.imshow(hand)
    plt.axis('off')
    plt.title('Hand')

    plt.show()

def create_dataset(config: Config):
    params = config.get_dataset_params()
    if config.dataset_type == 'RHD':
        return RHDDataset(config.dataset_root, config.dataset_split, config.dataset_to_tensor, **params)
    elif config.dataset_type == 'IsoGD':
        return IsoGDDataset(config.dataset_root, config.dataset_split, config.dataset_to_tensor, **params)
    elif config.dataset_type == 'Senz':
        return SenzDataset(config.dataset_root, config.dataset_split, config.dataset_to_tensor, **params)
    else:
        raise ValueError('不支持的数据集类型: {}'.format(config.dataset_type))

def create_segmentor(config: Config):
    params = config.get_seg_params()
    if config.seg_type == 'RDF':
        return RDFSegmentor(config.seg_model_path,**params)
    elif config.seg_type == 'ResNet':
        return ResNetSegmentor(config.seg_model_path, **params)
    else:
        raise ValueError('不支持的分割器类型: {}'.format(config.seg_type))
    

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='test', help='config file name')
    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO,
                        # level=logging.WARNING,
                        # level=logging.CRITICAL,
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[
                            # logging.FileHandler(f"Log/log {Utils.get_time_str()}.txt", encoding='utf-8'),
                            logging.StreamHandler()
                        ])
    
    # 配置matplot
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    config_file = str(args.config)
    config = Config(config_file)
    dataset = create_dataset(config)
    segmentor = create_segmentor(config)

    # Test.test_depth_feature(dataset)
    # Test.test_depth_feature_mask(dataset)
    # Test.test_rhd_dataset_mask_count(dataset)
    Test.test_rhd_dataset_depth_range(dataset)
    # Test.test_rhd_dataset_depth_max(dataset)
    # Test.test_senz_dataset(dataset)

    # Test.test_direct_method(dataset)

    # Test.test_iso_dataset_depth_range(dataset)

    # tree = Test.test_train_tree(dataset, segmentor)
    # tree = Test.test_train_tree_iterative(dataset, segmentor)
    # Test.test_one_tree_predict(dataset, segmentor, sample_idx=1266)
    # tree = Test.test_train_forest(dataset, segmentor)
    # Test.test_forest_predict(dataset, segmentor)
    # Test.test_one_tree_result(dataset, segmentor, 
    #                           force_train=False, 
    #                         #   force_predict=False, 
    #                           tree_count=200, predict_count=5)
    # Test.test_vary_max_depth(dataset, segmentor)