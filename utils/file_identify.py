# -*- coding: utf-8 -*-
"""
@Time    : 2023/10/25 11:32
@FileName: file_identify.py
@Project : atmospheric_correction
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""
import os


def infile_list(path, sensor):
    dirs = os.listdir(path)
    files = []
    for subdir in dirs:
        if sensor == "olcis3a" or sensor == "olcis3b":
            if ("_OL_1_EFR____" in subdir) and (".SEN3" in subdir):
                files.append(os.path.join(path, subdir))
        if sensor == "sdgsat1mii":
            if ("KX10_MII_" in subdir) and ("L4B" in subdir):
                filepath = os.path.join(path, subdir)
                if not os.path.isdir(filepath):
                    continue
                infile_temp = filepath + os.sep + subdir + "_ROI.tif"
                files.append(infile_temp)
        else:
            print("No given sensor ID!!!")
    print("***total {} images***:".format(files.__len__()), [os.path.basename(i) for i in files])
    return files
