# -*- coding: utf-8 -*-
"""
@Time    : 2023/10/8 11:00
@FileName: read.py
@Project : atmospheric_correction
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""
import os
from . import block_read
import warnings
import bs4
warnings.filterwarnings("ignore")


def read_xml(in_file_dir):
    file = os.path.join(in_file_dir, "xfdumanifest.xml")
    soup = bs4.BeautifulSoup(open(file), 'lxml')
    rows = int(soup.findAll('sentinel3:rows')[0].text)
    columns = int(soup.findAll('sentinel3:columns')[0].text)
    basename = os.path.basename(in_file_dir)
    time = basename.split("_")[7]
    year = time[0:4]
    month = time[4:6]
    day = time[6:8]
    hour = time[8:10]
    minute = time[10:12]
    # time = int(soup.findAll("sentinel3:receivingStartTime")[0].text)
    return rows, columns, year, month, day, hour, minute


def get(infile, blocksize=None):
    rows, columns, year, month, day, hour, minute = read_xml(infile)
    data_Iterator = block_read.ReadIterator(infile, blocksize=blocksize, rows=rows, columns=columns)
    num_443, num_490, num_520, num_555, num_670, nirs_num, nirl_num = 2, 3, 4, 5, 8, 11, 16,
    nwvis = 10
    red = num_670

    return [data_Iterator, year, month, day, num_443, num_490, num_520, num_555,
            num_670, nirs_num, nirl_num, nwvis, red, rows, columns]
