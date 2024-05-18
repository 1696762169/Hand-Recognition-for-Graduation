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
import time
import argparse
from typing import Literal

import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import logging
from tqdm import tqdm

from Config import Config
from PSO import PSOSolver
from Utils import Utils
from HandModel import HandModel
from Evaluation import Evaluation
from Dataset.RHD import RHDDataset
from Dataset.IsoGD import IsoGDDataset
from Dataset.Senz import SenzDataset
from Segmentation import ResNetSegmentor, RDFSegmentor
from Tracker import ThreeDSCTracker
import Test

# dataset_type: Literal['RHD', 'IsoGD', 'Senz'] = 'RHD'
dataset_type: Literal['RHD', 'IsoGD', 'Senz'] = 'Senz'
segmentor_type: Literal['RDF', 'ResNet'] = 'RDF'
tracker_type: Literal['3DSC'] = '3DSC'

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
    
def create_tracker(config: Config):
    params = config.get_track_params()
    if config.track_type == '3DSC':
        return ThreeDSCTracker(**params)
    else:
        raise ValueError('不支持的追踪器类型: {}'.format(config.track_type))

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # 解析命令行参数
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data', type=str, default='RHD', help='dataset type, default: RHD')
    # parser.add_argument('--seg', type=str, default='RDF ', help='segmentor type, default: RDF')
    parser.add_argument('--config', type=str, help='config file name')
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
    
    config_file = str(args.config) if args.config is not None else None
    config = Config(dataset_type=dataset_type, seg_type=segmentor_type, track_type=tracker_type, config_path=config_file)
    dataset = create_dataset(config)
    segmentor = create_segmentor(config)
    tracker = create_tracker(config)

    # Test.test_depth_feature(dataset)
    # Test.test_depth_feature_mask(dataset)
    # Test.test_rhd_dataset_mask_count(dataset)
    # Test.test_rhd_dataset_depth_range(dataset)
    # Test.test_rhd_dataset_depth_max(dataset)
    # Test.test_senz_dataset(dataset)

    # Test.test_senz_bilateral_filter(dataset)

    # Test.test_direct_method(dataset)
    # Test.test_resnet_predict(dataset, segmentor)

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

    Test.test_contour_extract(dataset, tracker)