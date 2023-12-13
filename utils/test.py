import numpy as np


import os.path

import h5py


def create(file):
    if os.path.exists(file):
        os.remove(file)
    ds = h5py.File(file, mode="a")
    navi_group = ds.create_group("Navigation data")
    geo_group = ds.create_group("Geophysical data")
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

    if data_name not in ds_group.keys():
        dset = ds_group.create_dataset(data_name, shape=chunk.shape, maxshape=(None, columns),
                                       chunks=True, dtype=chunk.dtype, compression="gzip")
    else:
        dset = ds_group[data_name]
    rows_ = chunk.shape[0]
    rows_new = rows_ext + rows_
    dset[rows_ext:rows_new] = chunk
    print(rows_ext, rows_new)
    return rows_new


def close(ds):
    ds.close()
    return

def new():
    file = "test.h5"
    ds = h5py.File(file, mode="a")
    ds_group = ds.create_group("Geophysical data")
    data_name="a"
    chunk=np.zeros(shape=(20,1000))+2
    dset = ds.create_dataset('X_train', shape=chunk.shape,  dtype=chunk.dtype, compression="gzip", chunks=True, maxshape=(None,1000))
    # dset = ds.create_dataset(data_name, maxshape=(40, 1000),
    #                          chunks=(2,None), dtype=chunk.dtype, compression="gzip")
    dset[0:20,:]=chunk
    dset.resize((dset.shape[0] + 20, 1000))
    dset[20:40,:]=chunk+2
    ds.close()

import os.path

import h5py


def create(file):
    if os.path.exists(file):
        os.remove(file)
    ds = h5py.File(file, mode="a")
    navi_group = ds.create_group("Navigation data")
    geo_group = ds.create_group("Geophysical data")
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
        ds_group.create_dataset(data_name, shape=chunk.shape, maxshape=(None, columns),
                                chunks=True, dtype=chunk.dtype, compression="gzip")
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


if __name__ == '__main__':
    ds, navi_group, geo_group = create('test2.h5')
    chunk = np.zeros(shape=(20, 1000)) + 2
    rows_ext = 0
    write(geo_group, "data", rows_ext, chunk, 1000)
    rows_ext += chunk.shape[0]
    chunk = np.zeros(shape=(8, 1000)) + 4
    write(geo_group, "data", rows_ext, chunk, 1000)
    rows_ext += chunk.shape[0]
    chunk = np.zeros(shape=(3, 1000)) + 4
    write(geo_group, "data", rows_ext, chunk, 1000)




