# -*- coding: utf-8 -*-
"""
@Time    : 2023/10/4 22:02
@FileName: esdist.py
@Project : atmospheric_correction
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""
import numpy as np


def esdist(day):
    return 1.00014 - 0.01671 * np.cos(2.0 * np.pi * (0.9856002831 * day - 3.4532868) / 360.0) - 0.00014 * np.cos(
        4.0 * np.pi * (0.9856002831 * day - 3.4532868) / 360.0)

# print(esdist(45))
