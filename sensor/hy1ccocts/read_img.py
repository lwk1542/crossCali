# -*- coding: utf-8 -*-
"""
@Time    : 2023/10/8 11:00
@FileName: read.py
@Project : atmospheric_correction
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""
import datetime
from . import block_read
import warnings
import h5py
warnings.filterwarnings("ignore")


def get(infile, blocksize: int=None)->list:
    ds = h5py.File(infile, mode="r")
    day_time = ds.attrs["Scene Center Time"].decode('UTF-8')
    dt = datetime.datetime.strptime(day_time, "%Y-%m-%dT%H:%M:%S")
    year, month, day, hour, minute = dt.year, dt.month, dt.day, dt.hour, dt.minute
    columns = ds.attrs["Pixels Per Scan Line"]
    (rows, columns) = ds["Navigation Data/" + "Latitude"].shape
    if blocksize > rows:
        blocksize = rows
    data_Iterator = block_read.ReadIterator(infile, blocksize=blocksize, rows=rows, columns=columns)
    num_443, num_490, num_520, num_555, num_670, nirs_num, nirl_num = 1, 2, 3, 4, 5, 6, 7
    nwvis = 8
    red = num_670

    return [data_Iterator, year, month, day, num_443, num_490, num_520, num_555,
            num_670, nirs_num, nirl_num, nwvis, red, rows, columns]
