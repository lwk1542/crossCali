# -*- coding: utf-8 -*-
"""
@Time    : 2023/11/23 17:42
@FileName: decomp.py
@Project : atmospheric_correction
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""
import glob
import os

def zip(infile):
    import zipfile
    # archive = tarfile.open(infile)
    # outpath = infile[infile.rfind("/") + 1:infile.rfind(".")]
    outpath = os.path.dirname(infile) + os.sep + os.path.splitext(os.path.basename(infile))[0]
    archive = zipfile.ZipFile(infile, 'r')  # 压缩文件位置
    zip_list = archive.namelist()  # 得到压缩包里所有文件
    print("解压缩...")
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    for f in zip_list:
        if not os.path.exists(os.path.join(outpath, f)):
            archive.extract(f, outpath)  # 循环解压文件到指定目录
    archive.close()  # 关闭文件，必须有，释放内存
    tif_a = outpath + os.sep + os.path.basename(outpath) + ".h5"
    print(os.path.basename(tif_a))
    return tif_a


def targz_hy1(infile):
    import tarfile
    # outpath = infile[infile.rfind("/") + 1:infile.rfind(".")]
    outpath = os.path.dirname(infile) + os.sep + os.path.splitext(os.path.basename(infile))[0]
    # outfile = outpath + os.sep + os.path.basename(outpath) + ".h5"
    # if os.path.exists(outpath):
    #     return outfile
    print("解压缩...")
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    archive = tarfile.open(infile)
    archive.extractall(outpath)
    archive.close()  # 关闭文件，必须有，释放内存
    # print(os.path.basename(outfile))
    outfile = glob.glob(outpath+os.sep+"*h5")[0]
    return outfile