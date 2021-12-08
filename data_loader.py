#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 01:11:06 2021

@author: ychen413
"""
# Library
# Standard library
import pathlib
import re
import random

# Third party library
import numpy as np

# Pytorch library
import torch
import torch.utils.data

# load project files
from config import DefaultConfig

# Dataset for DataLoader
class CustomTensorDataset(torch.utils.data.Dataset):
    """PUFid dataset: numpy array (num_data, 2556) tp tensor
    """
    def __init__(self, folder_path, filetype='.csv', delimiter=',', is_target_signal=False):
        dataset = []
        label = []
        label_signal = {}
        
        p = pathlib.Path(folder_path)
        for file in p.iterdir():
            if file.suffix == filetype:
                label_ = int(re.findall(r'[\d]+', file.name)[0]) - 1    # Make the label between 0 - 49 
                with open(file) as f:
                    data = np.loadtxt(f, delimiter=delimiter).transpose()
                    dataset.extend(data[2:])
                    label.extend(np.repeat(label_, len(data[2:])))
                    label_signal[label_] = data[1]
                    
        self.dataset = np.array(dataset)
        self.label = np.array(label)
        self.label_signal = label_signal
        self.is_target_signal = is_target_signal        
        
    def __getitem__(self, index):
        data = self.dataset[index]
        data = torch.from_numpy(data[None, :])
        
        if self.is_target_signal:
            target = self.label_signal[self.label[index]]
            target = torch.from_numpy(target[None, :])
        else:
            target = self.label[index]

        # data_ = (data - data.mean()) / data.std()    # Normalize with mean and std
        # data_ = (data - data.min()) / (data.max() - data.min())    # Normalize to [0 1]
        
        return data, target

    def __len__(self):
        return len(self.dataset)
    
# Load ground truth dataset
# def load_groundtruth(folder_path, filetype='.csv', delimiter=','):
#     ground_truth = []
#     label_ground = []
    
#     p = pathlib.Path(folder_path)
#     for file in p.iterdir():
#         if file.suffix == filetype:
#             label = int(re.findall(r'[\d]+', file.name)[0])
#             label_ground.append(label)
#             with open(file) as f:
#                 data = np.loadtxt(f, delimiter=delimiter).transpose()
#                 ground_truth.append(data[1])
    
#     ground_truth = torch.from_numpy(np.array(ground_truth))
#     label_ground = torch.from_numpy(np.array(label_ground))
    
#     return ground_truth, label_ground

def make_timeSeries():
    return np.arange(50, 305.6, 0.1)


def index_split(num_data, train_rate=0.8, seed=None):
    # Split data into train and test part
    # Output: index sets for training and testing dataset
    # Use dataset[index] to get the training / testing dataset
    ind = list(range(num_data))
    num_train = int(np.floor(num_data * train_rate))
    if seed != None:
        random.seed(seed)
    random.shuffle(ind)
    ind_train, ind_test = ind[:num_train], ind[num_train:]
    return ind_train, ind_test

# Import dataset and ground truth
# root = DefaultConfig.data_root
# dataset = CustomTensorDataset(root)
# ground_truth, label_ground = load_groundtruth(root)







    
    
    
    
    
    