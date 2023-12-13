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


def cloud_land_mask(lt, sza, F0):
    """
    """
    (rows_, columns_) = sza.shape
    mu = np.cos(np.deg2rad(sza)).reshape(rows_, columns_, 1)
    rhot = np.pi * lt / F0 / mu
    m1 = rhot[:, :, 1] > rhot[:, :, 2]
    m2 = rhot[:, :, 2] > rhot[:, :, 3]
    m3 = rhot[:, :, 3] > rhot[:, :, 4]
    m4 = rhot[:, :, 4] > rhot[:, :, 5]
    m5 = rhot[:, :, 5] > rhot[:, :, 6]
    z = m1 & m2 & m3 & m4 & m5
    for i in range(7):
        lt[:, :, i][~z] = np.nan
        lt[:, :, i][lt[:, :, i] > F0[0, 0, i]] = np.nan
    ret, binary = cv2.threshold(lt[:, :, 0], 0, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3))
    open1 = cv2.erode(binary, kernel, iterations=2)  # 腐蚀
    open1[np.isnan(open1)] = np.nan
    open1 = open1/open1
    open1[open1 <= 0] = np.nan
    open1[open1 > 260] = np.nan
    open1[~np.isnan(open1)] = 1
    # lt = lt * open1.reshape(self.rows_chunk, self.columns_chunk, 1)
    return lt * open1.reshape(rows_, columns_, 1)