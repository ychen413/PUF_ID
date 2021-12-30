#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 12:33:51 2021

@author: ychen413
"""

# Config

import numpy as np
import torch
# from torch.utils.data import Dataset, TensorDataset
# import torchvision
# import torchvision.transforms as transforms


class DefaultConfig(object):
    data_root = 'D:\Research_USA\python_project\PUF_id\DataSet\\'
    out_dir = '.\results'
    
    # General setting
    niter = 101
    batch_size = 128
    lr = 0.0015
    beta1 = 0.5
    
    workers = 12
    cuda = True
    ngpu = 1
    
    # For ResNet / CNN
    num_id = 50
    initial_kernel = 16
    kernel_multipler = 2
    
    # For U-net
    in_channels = (1, 64, 128, 256, 512, 1024)
    out_channels = (1024, 512, 256, 128, 64)
    out_num = 4
    
    
