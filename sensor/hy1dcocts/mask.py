# -*- coding: utf-8 -*-
"""
@Time    : 2023/11/1 10:27
@FileName: cloud_mask.py
@Project : atmospheric_correction
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""
import numpy as np
import cv2


def cloud_land_mask(rhos:np.ndarray)->np.ndarray:
    """
    """
    (rows_, columns_) = rhos[:, :, 0].shape
    # mu = np.cos(np.deg2rad(sza)).reshape(rows_, columns_, 1)
    # rhot = np.pi * lt / F0 / mu
    k = (rhos[:, :, 7]-rhos[:, :, 1])/(7-1)
    # y1 = k * (1 - 0) + rhos[:, :, 0]
    y2 = k * (2 - 1) + rhos[:, :, 1]
    y3 = k * (3 - 1) + rhos[:, :, 1]
    y4 = k * (4 - 1) + rhos[:, :, 1]
    y5 = k * (5 - 1) + rhos[:, :, 1]
    y6 = k * (6 - 1) + rhos[:, :, 1]
    # m1 = rhos[:, :, 1] > y1
    m2 = rhos[:, :, 2] > y2
    m3 = rhos[:, :, 3] > y3
    m4 = rhos[:, :, 4] > y4
    m5 = rhos[:, :, 5] > y5
    m6 = rhos[:, :, 6] > y6
    z1 = m2 & m3 & m4 & m5 & m6
    z2 = (~m2) & (~m3) & (~m4) & (~m5) & (~m6)
    z3 = rhos[:, :, 7] < rhos[:, :, 6]
    z = (z1 | z2) & z3
    # for i in range(7):
    rhos[:, :, 0][~z] = np.nan
        # lt[:, :, i][lt[:, :, i] > F0[0, 0, i]] = np.nan
    ret, binary = cv2.threshold(rhos[:, :, 0], 0, 255, cv2.THRESH_BINARY)
    kernel = np.ones((2, 2))
    open1 = cv2.erode(binary, kernel, iterations=1)  # 腐蚀
    open1[np.isnan(open1)] = np.nan
    open1 = open1/open1
    open1[open1 <= 0] = np.nan
    open1[open1 > 260] = np.nan
    open1[~np.isnan(open1)] = 1
    # lt = lt * open1.reshape(self.rows_chunk, self.columns_chunk, 1)
    return open1.reshape(rows_, columns_, 1)