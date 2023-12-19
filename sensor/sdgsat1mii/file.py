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
        _ = path + os.sep + subdir+os.sep+subdir+"_B.tif"
        # "A相机大部分处于耀光范围内，干脆不要了"
        if os.path.exists(_):
            files.append(_)
        else:
            continue
        # if ("_L4B_B.tif" not in subdir) or ("KX10_MII_" not in subdir):
        #     continue
    return files
