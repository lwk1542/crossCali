# -*- coding: utf-8 -*-
"""
@Time    : 2023/11/29 11:48
@FileName: files_search.py
@Project : atmospheric_correction
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""
import os


def search_(path: str):
    subdirs = os.listdir(path)
    files = []
    for subdir in subdirs:
        if ("FY3D_MERSI_GBAL_L1_" not in subdir) or ("_1000M_MS.HDF" not in subdir):
            continue
        _ = path + os.sep + subdir
        files.append(_)
    return files
