# -*- coding: utf-8 -*-
"""
@author: li wenkai
@contact: lwk1542@hotmail.com
@software: pycharm anaconda python3.7 
@file: general.py
@time: 2021/3/8 12:31
@desc:
"""
import os
import numpy.matlib
from scipy import interpolate

def get_filelist(path, idx, *args):
    """
    Args:
        path (): 文件检索的父目录
        idx (): 一个指示文件字符串，
        *args (): 任意多个指示文件的字符串
    Returns:
        检索文件：输入父目录和至少一个能够指示文件名的字符串，比如‘.hdf’等
    get_filelist(rayleigh_lut_path, 'rayleigh', 'iqu.hdf')
    """
    Filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            # 如果是需要包含任何一个字符串，则用any
            if all(idxi in filename for idxi in args + (idx,)):
                Filelist.append(os.path.join(home, filename))
    return Filelist


def get_filelistv2(idx, *args, path=None, mode='all'):
    """
    输入父目录和至少一个能够指示文件名的字符串，比如‘.hdf’等
    Args:
        path (): 文件检索的父目录
        mode (): 关键字检索模式，或（any）/且（all），any表示任何一个关键字符匹配就可以
        idx (): 一个指示文件字符串，
        *args (): 任意多个指示文件的字符串

    Returns:
        检索到的文件列表
    """
    import os
    Filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            # 如果是需要包含任何一个字符串，则用any
            # 这里是不区分大小写的
            if mode == 'all':
                if all(idxi.lower() in filename.lower() for idxi in args + (idx,)):
                    Filelist.append(os.path.join(home, filename))
            elif mode == 'any':
                if all(idxi.lower() in filename.lower() for idxi in args + (idx,)):
                    Filelist.append(os.path.join(home, filename))
            else:
                print('the mode parameter was wrong and should be "all","any",or nothing')
    return Filelist


def intper2(lat_=None, lon_=None, value=None, Lon=None, Lat=None):

    # 经度从-180----180的坐标转换为 0-360度
    Lon[Lon < 0] = Lon[Lon < 0] + 360
    lon_[lon_ < 0] = lon_[lon_ < 0] + 360

    # 网格化处理
    # llon = numpy.matlib.repmat(lon_, lat_.shape[0], 1)
    # llat = (numpy.matlib.repmat(lat_, lon_.shape[0], 1)).T

    return interpolate.griddata((lat_.flatten(), lon_.flatten()), value.flatten(), (Lat, Lon), method='linear')


def file_check(file_name):
    """
    如果有重复文件，命名加（1），（2）。。。
    Args:
        file_name ():

    Returns:

    """
    import os
    temp_file_name = file_name
    i = 1
    while i:
        # print(temp_file_name)
        # print(os.path.exists("static/" + temp_file_name))
        if os.path.exists(temp_file_name):
            idx=file_name.rfind('-')
            name=file_name[0:idx]
            suffix=file_name[idx:]
            # name, suffix = file_name.split('.')
            name += '(' + str(i) + ')'
            temp_file_name = name+suffix
            i = i+1
        else:
            return temp_file_name