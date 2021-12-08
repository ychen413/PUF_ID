#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 14:47:53 2021

@author: ychen413
"""

from __future__ import print_function
import os
import time
import timeit
import random
import numpy as np
import matplotlib.pyplot as plt
#import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as Data
import torchvision
from torchvision import datasets, models, transforms
import torchvision.utils as vutils
#from torch.autograd import Variable

### load project files
import models
from models import weights_init
from config import DefaultConfig
import data_loader
# from data_loader import dataset, ground_truth, label_ground

if __name__ ==  '__main__':    
    #%% Set Cuda
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    #%% Load configurations
    opt = DefaultConfig()
    
    try:
        os.makedirs(opt.out_dir)
    except OSError:
        pass
    
    root = opt.data_root
    
    ngpu = opt.ngpu
    n_worker = opt.workers
    
    n_epoch = opt.niter
    n_batch = opt.batch_size
    learning_rate = opt.lr
    ch_in = opt.in_channels
    ch_out = opt.out_channels
    n_fig_show = opt.out_num
    
    #%% Set dataloader
    # Import dataset and ground trutht
    dataset = data_loader.CustomTensorDataset(root, is_target_signal=True)
    
    dataloader = Data.DataLoader(
            dataset=dataset,
            batch_size=n_batch,
            shuffle=True,
            num_workers=n_worker,
            )
    
    # Use to plot the signals
    ts = data_loader.make_timeSeries()
    
    # ind_train, ind_test = data_loader.index_split(len(dataset), train_rate=0.8, seed=None)
    # sampler_train = Data.SubsetRandomSampler(ind_train)
    # sampler_test = Data.SubsetRandomSampler(ind_test)
    
    # train_loader = Data.DataLoader(
    #         dataset=dataset,
    #         sampler= sampler_train,
    #         batch_size=n_batch,
    #         num_workers=n_worker,
    #         )
    
    # test_loader = Data.DataLoader(
    #         dataset=dataset,
    #         sampler=sampler_test,
    #         batch_size=n_batch,
    #         num_workers=n_worker,
    #         )
    
    #%% Import model
    # net = models.CNN_net(num_id=n_class, initial_kernel=n_kernel, multipler=n_mul)
    net = models.Unet(ch_encoder=ch_in, ch_decoder=ch_out, retain_dim=True)
    net.double()
    # print(net)
    
    #%% Main: Training and Testing
    
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-5)   # optimize all cnn parameters
    loss_func = nn.MSELoss()            # The Loss is the lower the better
                                        # NLLLoss: should return log_softmax(x) in the output
    
    result = np.zeros(2)
    min_loss = 999
    
    start = timeit.default_timer()
    for epoch in range(n_epoch):
        loss_sum = 0
        for step, (x, y) in enumerate(dataloader):   # gives batch data, normalize x when iterate train_loader
                                                           # x: signal, y: label
            # x = amp_gaussian(x, is_training=True, std=0.4)
            out = net(x)            # cnn output
            
            # np.set_printoptions(precision=3, suppress=True)
            # print('Step %d: '%step, output.data.numpy())
            loss = loss_func(out, y)          # MSE loss
            optimizer.zero_grad()             # clear gradients for this training step
            loss.backward()                   # backpropagation, compute gradients
            optimizer.step()                  # apply gradients
            
            loss_sum += (loss.data * len(x))
        
        total_loss = loss_sum / len(dataset)
        print('Epoch %d | Total Loss = %6.4f' % (epoch, total_loss))
        result = np.vstack((result, np.array([epoch, total_loss])))
        
        # Find best result:        
        if total_loss < min_loss:
            e = epoch
            net_best = net.state_dict()
            
        if epoch % 10 == 0:
            ind_select = np.random.randint(len(x), size=4)
            for ind in ind_select:
                plt.figure()
                plt.plot(ts, y.data[ind].view(-1).numpy(), 'r', label='Ground')
                plt.plot(ts, out.data[ind].view(-1).numpy(), 'b', label='Unet')
                plt.xlabel('Time (sec)')
                plt.ylabel('Amplitude')
                plt.legend(loc='lower right')
    
    stop = timeit.default_timer()
    t_run = stop - start
    print('Running time: %.4f (sec) ~ %d min %d sec'%(t_run, int(t_run/60), int(t_run%60)))

                                    
                                    
                                    
                                    
                                    