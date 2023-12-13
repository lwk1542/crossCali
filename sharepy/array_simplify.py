# -*- coding: utf-8 -*-
"""
@author: li wenkai
@contact: lwk1542@hotmail.com
@software: pycharm anaconda python3.7 
@file: array_simplify.py
@time: 2021/3/21 19:11
@desc:为了提高计算效率，这个函数将nan踢出数组组成新的数组，计算完成后，又恢复数组的形状
"""
import numpy as np


def delete_nan(array=None):
    """
    一个带有nan的数组，首先给数组加两个层：行号和列号，然后删除位置的数据
    """
    rows = np.ones_like(array[:, :, 0]) * (np.arange(0, array.shape[0], 1).reshape(array[:, :, 0].shape[0], 1))
    columns = np.ones_like(array[:, :, 0]) * np.arange(0, array.shape[1], 1).T
    new_array = np.empty(shape=(array.shape[0], array.shape[1], array.shape[2]+2))
    new_array[:, :, 0:array.shape[2]] = array
    new_array[:, :, -2] = rows
    new_array[:, :, -1] = columns
    new_1=np.zeros(shape=(array.shape[2] + 2, array.shape[0]*array.shape[1]))
    for i in range(new_array.shape[2]):
        new_1[i, :] = new_array[:, :, i].reshape(1, -1)
    new_2 = new_1[:, ~np.isnan(new_1).any(axis=0)]
    # new_2的最后两行代表数据在原数组中的位置，其它每行代表一个波段
    return new_2


def recover_nan(new_array=None, rows: int = None, columns: int = None):
    new_3 = np.full(shape=(rows, columns, new_array.shape[0] - 2), fill_value=np.nan)
    for i in range(new_array.shape[0]-2):
        new_3[list(new_array[-2, :].astype(int)), list(new_array[-1, :].astype(int)), i] = new_array[i, :]
    return new_3


if __name__ == "__main__":
    a=np.empty(shape=(3,4,2))
    a[:,:,0]=np.random.randint(0, 10, size=[3, 4]) * 1.
    a[:, :, 1] = np.random.randint(10, 20, size=[3, 4]) * 1.
    a[0, 0,0], a[1,1, 0], a[2,3,1] = np.nan,np.nan,np.nan
    new_array=delete_nan(array=a)
    recover_nan(rows=a.shape[0], columns=a.shape[1], new_array=new_array)
