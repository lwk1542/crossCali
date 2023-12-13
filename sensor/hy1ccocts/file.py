# -*- coding: utf-8 -*-
"""
@Time    : 2023/11/29 11:48
@FileName: files_search.py
@Project : atmospheric_correction
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""
import os, tarfile


def search_(path: str):
    subdirs = os.listdir(path)
    files = []
    for subdir in subdirs:
        if "H1C_OPER_OCT_L1A" not in subdir:
            continue
        infile = path + os.sep + subdir
        outpath = path + os.sep + subdir.split(sep=".")[0]
        _ = outpath+os.sep+subdir.split(sep=".")[0] + ".h5"
        if _ in files:
            continue
        if os.path.exists(_):
            files.append(_)
            continue
        if "_10.tar.gz" not in subdir:
            continue
        if not os.path.exists(outpath):
            os.mkdir(outpath)
        print("解压缩...{0}".format(subdir))
        archive = tarfile.open(infile)
        archive.extractall(outpath)
        archive.close()  # 关闭文件，必须有，释放内存
        files.append(_)
    return files