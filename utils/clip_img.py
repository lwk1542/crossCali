# -*- coding: utf-8 -*-
"""
@Time    : 2022/11/13 16:03
@FileName: clip_img.py
@Project : git_repository
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""
def img_clip():
    # 范围裁剪
    loc = np.where((south < lat_grid) & (lat_grid <north) & (west < lon_grid) & (lon_grid < east))
    # if loc[0].size < predefine.thresholds().pixels_num:  # 不到最小像元数量直接不计算了
    #     return ()
    up, low, left, right = np.min(loc[0]), np.max(loc[0]), np.min(loc[1]), np.max(loc[1])
    lat_grid = lat_grid[up:low, left:right]
    lon_grid = lon_grid[up:low, left:right]
    sza = sza[up:low, left:right]
    vza = vza[up:low, left:right]
    saa = saa[up:low, left:right]
    vaa = vaa[up:low, left:right]
    vza[vza > 88] = 88
    vza[vza < 0] = 0
    sza[sza > 88] = 88
    sza[sza < 0] = 0
    data = data[up:low, left:right, :]


if all([north, south, west, east]):

    img_clip([sza, vza, saa, vaa, lat_grid, lon_grid, data])