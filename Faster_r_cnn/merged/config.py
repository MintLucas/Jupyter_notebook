#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 15:32:23 2018

@author: jon-liu
"""

import numpy as np

class Config():
    featureMap_size = [8, 8]
    scales = [4, 8, 16]
    ratios = [0.5, 1, 2]
    rpn_stride = 8
    anchor_stride = 1
    train_rois_num = 100
    image_size = [64, 64]
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    num_before_nms = 300
    max_gt_obj = 30
    num_proposals_train = 21
    num_proposals_ratio = 0.333
    batch_size = 20
    
    