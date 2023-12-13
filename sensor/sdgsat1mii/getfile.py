# -*- coding: utf-8 -*-
"""
@Time    : 2023/4/3 15:39
@FileName: getfile.py
@Project : git_repository
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""
import os


def getfile(inpath, date:str):
    dirs = os.listdir(inpath)
    print("total {} images:".format(dirs.__len__()), dirs)
    filepath=[]
    for subdir in dirs:
        filepath_temp = inpath + os.sep + subdir
        if not os.path.isdir(filepath_temp):
            continue
        if ("KX10_MII_"+date not in subdir) or ("L4A" not in subdir):
            continue
        filepath.append(filepath_temp)
    print("SDGSAT-1 MII files:", filepath)
    return filepath