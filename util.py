# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 16:13:47 2021

@author: user
"""

import numpy as np
import pathlib
import matplotlib.pyplot as plt
import os

def read_data(file_path, delimiter=',', transpose = 'True'):
    with open(file_path) as f:
        dataset = np.loadtxt(f, delimiter=',')
        
    if transpose:
        dataset = dataset.transpose()
        
    return dataset

def load_data(folder_path, filetype='.csv', delimiter=',', transpose='True'):
    p = pathlib.Path(folder_path)
    for filename in p.iterdir():
        if filename.suffix == filetype:
            with open(filename) as f:
                dataset = np.loadtxt(f, delimiter=delimiter)
        
        if transpose:
            dataset = dataset.transpose()
        
        return dataset
                

##

file_path = 'D:\Research_USA\python_project\PUFid\dataset_PUFid\\'
file_name = 'Device1-1.csv'

# p = pathlib.Path(file_path)
# for filename in p.iterdir():
#     if filename.suffix == '.csv':
#         with open(filename) as f:
#             dataset = np.loadtxt(f, delimiter=',').transpose()
#             # plt.figure()
#             # plt.plot(dataset[0, :], dataset[1, :])
            
data = read_data(f'{file_path}{file_name}')
for line in data[1:21]:
    plt.figure()
    plt.plot(data[0], line)