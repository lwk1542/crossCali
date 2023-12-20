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
import tarfile


def search_(path: str):
    # "返回文件夹"
    subdirs = os.listdir(path)
    files = []
    for subdir in subdirs:
        if "LC08_L1TP" not in subdir:   # 文件名不对
            continue
        filedir = path + os.sep + subdir
        if filedir in files:  # 已经添加
            continue
        if os.path.exists(filedir) and os.path.isdir(filedir): # 添加
            files.append(filedir)
            continue
        if "_T1.tar" in subdir:  # 是压缩文件
            filedir0 = os.path.splitext(filedir)[0]
            if filedir0 in files:  # 已经添加
                continue
            else:
                if not os.path.exists(filedir0):
                    os.mkdir(filedir)
                print("解压缩...{0}".format(subdir))
                archive = tarfile.open(filedir)
                archive.extractall(filedir0)
                archive.close()  # 关闭文件，必须有，释放内存
                files.append(filedir0)
        else:
            pass
    return files
