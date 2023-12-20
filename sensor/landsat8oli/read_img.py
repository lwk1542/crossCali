# -*- coding: utf-8 -*-
"""
@Time    : 2023/10/25 17:45
@FileName: read_img2.py
@Project : atmospheric_correction
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""
import os
from . import block_read
from . import landsat_metadata
import warnings
warnings.filterwarnings("ignore")

def meta_xml(file):
    datetime_obj, gains, offsets, rows, columns = landsat_metadata.mtl(file=file)
    datetime = datetime_obj.strftime("%Y-%m-%dT%H:%M:%S.%f")  # 2023-04-10T15:39:26.20453
    year, month, day, hour, minute, second = \
        int(datetime[0:4]), int(datetime[5:7]), int(datetime[8:10]), int(datetime[11:13]), int(datetime[14:16]), float(datetime[17:])

    return year, month, day, hour, minute, columns, rows


def get(infile, blocksize=None):
    meta_file_path = infile + os.sep + os.path.basename(infile) + "_MTL.xml"
    year, month, day, hour, minute, columns, rows = meta_xml(meta_file_path)

    data_Iterator = block_read.ReadIterator(infile, blocksize=blocksize, rows=rows, columns=columns)
    num_443, num_490, num_520, num_555, num_670, nirs_num, nirl_num = 0, 1, None, 2, 3, 4, 5,
    nwvis = 4
    red = num_670
    return [data_Iterator, year, month, day, num_443, num_490, num_520, num_555,
            num_670, nirs_num, nirl_num, nwvis, red, rows, columns]

