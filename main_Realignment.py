#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 18:05:40 2021

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
n_class = opt.num_id
n_kernel = opt.initial_kernel
n_mul = opt.kernel_multipler
n_epoch = opt.niter
n_worker = opt.workers
n_batch = opt.batch_size
learning_rate = opt.lr

#%% Set dataloader
# Import dataset and ground truth
dataset = data_loader.CustomTensorDataset(root)

ind_train, ind_test = data_loader.index_split(len(dataset), train_rate=0.8, seed=None)
sampler_train = Data.SubsetRandomSampler(ind_train)
sampler_test = Data.SubsetRandomSampler(ind_test)

train_loader = Data.DataLoader(
        dataset=dataset,
        sampler= sampler_train,
        batch_size=n_batch,
        num_workers=n_worker,
        )

test_loader = Data.DataLoader(
        dataset=dataset,
        sampler=sampler_test,
        batch_size=n_batch,
        num_workers=n_worker,
        )

#%% Import model
# net = models.CNN_net(num_id=n_class, initial_kernel=n_kernel, multipler=n_mul)
net = models.ResNet1d(num_id=n_class)
net.double()
# print(net)

#%% Main: Training and Testing

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()   # CrossEntropyLoss already includes softmax in the calculation
                                    # The Loss is the lower the better
                                    # NLLLoss: should return log_softmax(x) in the output

result = np.zeros(3)
# pred_y_ = np.empty(l_test.size(0))
acc_ = 0

start = timeit.default_timer()
for epoch in range(n_epoch):
    for step, (x, y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
                                                       # x: signal, y: label
        # x = amp_gaussian(x, is_training=True, std=0.4)
        out = net(x)            # cnn output
        
#        np.set_printoptions(precision=3, suppress=True)
#        print('Step %d: '%step, output.data.numpy())
        loss = loss_func(out, y.long())   # cross entropy loss
        optimizer.zero_grad()             # clear gradients for this training step
        loss.backward()                   # backpropagation, compute gradients
        optimizer.step()                  # apply gradients

    for step, (x, y) in enumerate(test_loader):   # x: signal, y: label
        score = 0
        total = 0
        
        test_out = net(x)
        
        np.set_printoptions(precision=3, suppress=True)
#        print(F.softmax(test_output.data, dim=1).numpy())

        pred_y = torch.max(test_out, 1)[1].data.numpy()
#        print(pred_y)
        # acc = float((pred_y == y.numpy()).astype(int).sum()) / float(y.size(0))
        score += float((pred_y == y.numpy()).astype(int).sum())
        total += float(y.size(0))
        
    acc = score / total
    print('Epoch: ', epoch)
    print('Train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.4f' % acc)
    
    result = np.vstack((result, np.array([epoch, loss.data.numpy(), acc])))
    
    # Find best result:        
    if acc > acc_:
        ans_y = y.numpy()
        pred_y_ = pred_y
        acc_ = acc
        # Transfer Learning
#            net1_best = net1.state_dict()

stop = timeit.default_timer()
t_run = stop - start

#%% Plot result
result_ = result[1:]
acc_max = np.max(result_[:, 2])

print('optimal value: %.4f (round %d)' % (acc_max, np.argmax(result_[:, 2])))
print('Running time: %.4f (sec) ~ %d min %d sec'%(t_run, int(t_run/60), int(t_run%60)))

plt.figure()
plt.plot(result_[:, 0], result_[:, 1], label='ResNet')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss')
plt.legend(loc='upper right')

plt.figure()
plt.plot(result_[:, 0], result_[:, 2], label='ResNet')
#plt.ylim(0.6, 1)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.legend(loc='lower right')

#%% Save the result
np.savetxt('%s/C_ResNet1d_01.csv' % (opt.out_dir), result_, fmt='%6.4f', delimiter=',')
np.savetxt('%s/C_ResNet1d_best.csv' % (opt.out_dir), (ans_y, pred_y_), fmt='%d', delimiter=',')
# Currently only save the answer of last batch

