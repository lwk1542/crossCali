# -*- coding: utf-8 -*-
"""
@author: li wenkai
@contact: lwk1542@hotmail.com
@software: pycharm anaconda python3.7 
@file: H1B.py
@time: 2021/7/5 15:01
@desc:
"""
import numpy as np


def smooth(arr, win=[]):
    for i in arr.shape[0]:
        for j in arr.shape[1]:
            arr[i, j] = np.mean(arr[i - win[0]:i + win[0], j - win[1], j + win[1]])

    return
