#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 12:33:13 2021

@author: ychen413
"""

# library
# standard library
from collections import OrderedDict

# Pytorch library
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
#torch.manual_seed(1)    # reproducible

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    pass

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
        
# class CNN_Block
class Block_Resnet(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, num_layer=1):
        super().__init__()  # In python, super(block, self).__init__() = super().__init__() 

        self.conv1 = nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channel, out_channel, kernel_size=kernel_size, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channel)
        self.fit_kernels = nn.Conv1d(in_channel, out_channel, kernel_size=1, bias=False)
        self.pool = nn.MaxPool1d(2)
        
    def forward(self, x):
        residual = self.fit_kernels(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)) + residual)
        x = self.pool(x)
        return x
        

class ResNet1d(nn.Module):
    # Classification
    def __init__(self, num_id, ch=(16, 32, 32, 64, 64)):
        super().__init__()
        
        self.conv1 = nn.Conv1d(1, ch[0], kernel_size=5)
        self.bn1 = nn.BatchNorm1d(ch[0])
        self.relu = nn.ReLU(inplace=True)
        self.mp1 = nn.MaxPool1d(2)
        
        self.blocks = nn.ModuleList([Block_Resnet(ch[i], ch[i+1]) for i in range(len(ch)-1)])
        
        self.linear1 = nn.Linear(ch[-1] * 79, 512) # 256*159=40,704
        self.dp1 = nn.Dropout(p=0.2)
        self.linear2 = nn.Linear(512, num_id)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.mp1(self.relu(self.bn1(self.conv1(x))))
        
        for bk in self.blocks:
            x = bk(x)
        
        x = x.view(x.size(0), -1)
        x = self.dp1(self.relu(self.linear1(x)))
        x = self.linear2(x)
        return x


# class CNN_net(nn.Module):
#     # Classification: Recognize which id belongs to
#     def __init__(self, num_id, initial_kernel=16, multipler=2):
#         super(CNN_net, self).__init__()
        
#         self.main = nn.Sequential(OrderedDict([
#                 ('conv11', nn.Conv1d( 1, initial_kernel, 3)),
#                 ('relu11', nn.ReLU()),
#                 ('conv12', nn.Conv1d(initial_kernel, initial_kernel * multipler, 3)),
#                 ('relu12', nn.ReLU()),
#                 ('maxpool1', nn.MaxPool1d(2)),
#                 ('conv21', nn.Conv1d(initial_kernel * multipler, initial_kernel * multipler**2, 3)),
#                 ('relu21', nn.ReLU()),
#                 ('conv22', nn.Conv1d(initial_kernel * multipler**2, initial_kernel * multipler**3, 3)),
#                 ('relu22', nn.ReLU()),
#                 ('maxpool2', nn.MaxPool1d(2)),
#                 ('conv31', nn.Conv1d(initial_kernel * multipler**3, initial_kernel * multipler**4, 3)),
#                 ('relu31', nn.ReLU()),
#                 ('conv32', nn.Conv1d(initial_kernel * multipler**4, initial_kernel * multipler**5, 3)),
#                 ('relu32', nn.ReLU()),
#                 ('maxpool3', nn.MaxPool1d(2)),
#                 ('flatten', Flatten()),
#                 ('linear1', nn.Linear(initial_kernel * multipler**5 * 316, initial_kernel * multipler**5)),
#                 ('dropoutL1', nn.Dropout(p=0.2)),
#                 ('reluL1', nn.ReLU()),
#                 ('linear2', nn.Linear(initial_kernel * multipler**5, initial_kernel * multipler**4)),
#                 ('dropoutL2', nn.Dropout(p=0.2)),
#                 ('reluL2', nn.ReLU()),
#                 ('linear3', nn.Linear(initial_kernel * multipler**4, num_id))
#                 ]))

#     def forward(self, x):
#         x = self.main(x)
#         return x


#%% U-Net
class Block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3):
        super().__init__()  # In python, super(block, self).__init__() = super().__init__() 

        self.conv1 = nn.Conv1d(in_channel, out_channel, kernel_size)
        # self.BN = nn.BatchNorm1d(out_channel)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv1d(out_channel, out_channel, kernel_size)
        
    def forward(self, x):
        return self.relu(self.conv2(self.relu((self.conv1(x)))))

class Encoder(nn.Module):
    def __init__(self, ch=(1, 64, 128, 256, 512, 1024)):
        # (batch_size, 1, 64) - ch -> (batch_size, 1024, 152)
        super().__init__()
        
        self.enc_blocks = nn.ModuleList([Block(ch[i], ch[i+1]) for i in range(len(ch)-1)])
        self.maxpool = nn.MaxPool1d(2)
        
    def forward(self, x):
        features = []   # Concate when using ResNet or Unet
        for block in self.enc_blocks:
            x = block(x)
            features.append(x)
            x = self.maxpool(x)
        return features

class Decoder(nn.Module):
    def __init__(self, ch=(1024, 512, 256, 128, 64)):
        # (batch_size, 1024, 152) - ch -> (batch_size, 1, )
        super().__init__()
        
        self.num_ch = len(ch)
        self.deconv = nn.ModuleList([nn.ConvTranspose1d(ch[i], ch[i+1], 2, 2) for i in range(len(ch)-1)])
        self.dec_blocks = nn.ModuleList([Block(ch[i], ch[i+1]) for i in range(len(ch)-1)])
        
    def forward(self, x, enc_features):
        for ind in range(self.num_ch-1):
            x = self.deconv[ind](x)
            encf = self.crop(enc_features[ind], x)
            x = torch.cat([x, encf], dim=1)
            x = self.dec_blocks[ind](x)
        return x

    def crop(self, target_obj, standard_obj):
        _, _, len_s = standard_obj.shape
        _, _, len_t = target_obj.shape
        return target_obj[:, :, (len_t - len_s) // 2: (len_t + len_s) // 2]

class Unet(nn.Module):
    def __init__(self, ch_encoder=(1, 64, 128, 256, 512, 1024), 
                       ch_decoder=(1024, 512, 256, 128, 64),
                       num_class=1, retain_dim=False, out_length=2556):
        super().__init__()
        
        self.encoder = Encoder(ch_encoder)
        self.decoder = Decoder(ch_decoder)
        self.head = nn.Conv1d(ch_decoder[-1], num_class, 1)
        self.retain_dim = retain_dim
        self.resize_len = out_length
        
    def forward(self, x):
        enc_features = self.encoder(x)
        out = self.decoder(enc_features[-1], enc_features[:-1][::-1])
        out = self.head(out)
        
        if self.retain_dim:
            out = F.interpolate(out, self.resize_len)
        
        return out



