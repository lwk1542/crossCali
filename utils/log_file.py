# -*- coding: utf-8 -*-
"""
@Time    : 2023/10/1 15:07
@FileName: log_file.py
@Project : atmospheric_correction
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""


def log_record(record, logfile):
    print(record)
    logfile.write(record)
    return logfile