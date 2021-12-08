# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 14:20:51 2021

@author: user
"""

def hammingDistance(x: int, y: int) -> int:
    x = deci_to_bin(x)
    y = deci_to_bin(y)
    x, y = padding(x, y)
    hamming_distance = 0
    for ind in list(range(len(x))):
        if x[ind] != y[ind]:
            hamming_distance += 1
    return hamming_distance
    
def deci_to_bin(x: int) -> str:
    # not reversed
    xbin = []
    while x >= 2:
        if (x / 2).is_integer():
            xbin.append('0')
        else:
            xbin.append('1')
        x = x // 2
    xbin.append('1')
    return xbin

def padding(x, y):
    num_fill = len(x) - len(y)
    pd = '0' * abs(num_fill)
    if num_fill > 0:
        y.extend(pd) 
    if num_fill < 0:
        x.extend(pd)
    return x, y
            