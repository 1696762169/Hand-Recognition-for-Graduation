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
#_________#_______####_______####_____________#
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
from Tracker import ThreeDSCTracker, LBPTracker
from Classification import DTWClassifier
import Test
import Preprocess

dataset_type: Literal['RHD', 'IsoGD', 'Senz'] = 'RHD'
# dataset_type: Literal['RHD', 'IsoGD', 'Senz'] = 'Senz'
# segmentor_type: Literal['RDF', 'ResNet'] = 'RDF'
segmentor_type: Literal['RDF', 'ResNet'] = 'ResNet'
tracker_type: Literal['3DSC', 'LBP'] = '3DSC'
# tracker_type: Literal['3DSC', 'LBP'] = 'LBP'
classifier_type: Literal['DTW'] = 'DTW'

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
        return RDFSegmentor(config.seg_model_path, config.seg_roi_detection, config.seg_roi_extended, **params)
    elif config.seg_type == 'ResNet':
        return ResNetSegmentor(config.seg_model_path, config.seg_roi_detection, config.seg_roi_extended, **params)
    else:
        raise ValueError('不支持的分割器类型: {}'.format(config.seg_type))
    
def create_tracker(config: Config):
    params = config.get_track_params()
    if config.track_type == '3DSC':
        return ThreeDSCTracker(**params)
    elif config.track_type == 'LBP':
        return LBPTracker(**params)
    else:
        raise ValueError('不支持的追踪器类型: {}'.format(config.track_type))

def create_classifier(config: Config):
    params = config.get_cls_params()
    if config.cls_type == 'DTW':
        return DTWClassifier(**params)
    else:
        raise ValueError('不支持的分类器类型: {}'.format(config.cls_type))

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
    config = Config(dataset_type=dataset_type, seg_type=segmentor_type, track_type=tracker_type, cls_type=classifier_type, config_path=config_file)
    dataset = create_dataset(config)
    segmentor = create_segmentor(config)
    tracker = create_tracker(config)
    classifier = create_classifier(config)

    # Test.test_depth_feature(dataset)
    # Test.test_depth_feature_mask(dataset)
    # Test.test_rhd_dataset_mask_count(dataset)
    # Test.test_rhd_dataset_depth_range(dataset)
    # Test.test_rhd_dataset_depth_max(dataset)

    # Test.test_senz_dataset()
    # Test.test_senz_preprocessing()
    # Test.test_senz_combine_filter()
    # Test.test_senz_bilateral_filter()

    # Test.test_direct_method(dataset)
    # Test.test_resnet_predict(dataset, return_prob=False)
    
    # Test.test_predict_roi(dataset, segmentor)
    # Test.test_segement_with_roi(dataset, segmentor, return_prob=True)
    # Test.test_segment_result()
    # Test.test_segment_all_result(dataset, show_good=False, show_bad=True)

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

    # Test.test_contour_extract(dataset, tracker)
    # Test.test_simplify_contour(dataset, tracker)
    # Test.test_descriptor_features(dataset, tracker)
    # Test.test_feature_direction(dataset)

    # Test.test_lbp_features(dataset, segmentor)

    Test.test_feature_invariance(tracker)

    feature_dir = "./Dataset/Features"
    dtw_dir = "./SegModel"
    feature_name = type(tracker).__name__
    # Preprocess.calculate_features(dataset, segmentor, tracker, feature_dir)
    # Preprocess.calculate_features_distance(dataset, classifier, feature_name, feature_dir, dtw_dir)

    # Test.test_feature_load(dataset, tracker, classifier, feature_dir)
    # Test.test_feature_effect(feature_name, dtw_dir)
    # Test.test_fastdtw_speed(dataset, feature_dir)

    # Test.test_dtw_distance(dataset, tracker, classifier)
    # Test.test_custom_fastdtw(dataset, feature_dir)

    # Test.test_classify_result()