# -*- coding: utf-8 -*-
"""
@Time    : 2023/10/10 14:55
@FileName: create_outfile.py
@Project : atmospheric_correction
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""
import os.path
import datetime
import h5py
import numpy as np


def create(outfile):
    if os.path.exists(outfile):
        os.remove(outfile)
    ds = h5py.File(outfile, mode="a")
    ds.attrs["product time"] = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    ds.attrs["method"] = "seadas l2gen"
    # ds.attrs["author"] = "Li Wenkai"
    ds.attrs["email"] = "liwenkai@scsio.ac.cn"
    ds.attrs["institution"] = "south china sea institute of oceanology, CAS"
    navi_group = ds.create_group("navigation_data")
    geo_group = ds.create_group("geophysical_data")
    return ds, navi_group, geo_group


def write(ds_group, data_name, rows_ext, chunk, columns):
    """
    不断向h5数据集写入新的行
    Args:
        ds_group: 组
        data_name: 数据名
        rows_ext: 已经写入数据的行数，初始为0，写入一次，更新一次
        chunk: 要写入的数据
        columns: 数据的总列数
    Returns:
        已经写入的行数

    """

    if data_name not in ds_group:
        ds_group.create_dataset(data_name, shape=(0, 0), maxshape=(None, columns),
                                chunks=True, dtype=float, compression="gzip")
    dset = ds_group[data_name]
    rows_ = chunk.shape[0]
    rows_new = rows_ext + rows_
    dset.resize((dset.shape[0] + chunk.shape[0], columns))
    dset[rows_ext:rows_new, :] = chunk
    # print(rows_ext, rows_new)
    return rows_new


def close(ds):
    ds.close()
    return

