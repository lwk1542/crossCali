# -*- coding: utf-8 -*-
"""
@Time    : 2023/10/8 11:00
@FileName: read.py
@Project : atmospheric_correction
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
          feng yun-3d mersi
"""
import datetime
from . import block_read
import warnings
import h5py
warnings.filterwarnings("ignore")


def get(infile: str, blocksize: int) -> list:
    ds = h5py.File(infile, mode="r")
    if ds.attrs["Day Or Night Flag"] == "N":
        print("Night imagery")
        return ["night imagery"]
    day_time_begin = ds.attrs["Observing Beginning Date"].decode('UTF-8') + "T" + ds.attrs[
        "Observing Beginning Time"].decode('UTF-8')
    day_time_end = ds.attrs["Observing Ending Date"].decode('UTF-8') + "T" + ds.attrs["Observing Ending Time"].decode(
        'UTF-8')
    dt1 = datetime.datetime.strptime(day_time_begin, "%Y-%m-%dT%H:%M:%S.%f")
    dt2 = datetime.datetime.strptime(day_time_end, "%Y-%m-%dT%H:%M:%S.%f")
    delta = (dt2 - dt1) / 2.
    dt = dt1 + delta
    year, month, day, hour, minute = dt.year, dt.month, dt.day, dt.hour, dt.minute

    rows = ds.attrs["Scan_Line_number"][0]
    columns = ds.attrs["Pixels_per_Scan"][0]
    # (rows, columns) = ds["Geophysical Data/DN_412"].shape
    if blocksize > rows:
        blocksize = rows
    data_Iterator = block_read.ReadIterator(infile, blocksize=blocksize, rows=rows, columns=columns)
    num_443, num_490, num_520, num_555, num_670, nirs_num, nirl_num = 1, 2, 3, 3, 4, 6, 7
    # bands_name = [412, 443, 490, 555, 670, 709, 746, 865, 905, 936, 940, 1030]
    nwvis = 12
    red = num_670

    return [data_Iterator, year, month, day, num_443, num_490, num_520, num_555,
            num_670, nirs_num, nirl_num, nwvis, red, rows, columns]
