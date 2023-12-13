# -*- coding: utf-8 -*-
"""
@Time    : 2022/11/3 17:44
@FileName: read_sdgsat1mii.py
@Project : git_repository
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""
import os
import numpy as np
from readtif import GdalReadTifIterator
from xml.etree import ElementTree as ET
import glob
import warnings
warnings.filterwarnings("ignore")


def calib_xml(xmlfile):
    # xmlfile = "L4A.calib.xml"
    # f = open(file, "r", encoding="gb2312") # gb2312格式不好用
    # datasource = f.read()
    # per = ET.parse(datasource)
    gains = np.array([0.051560133, 0.036241353, 0.023316835, 0.015849666, 0.016096381, 0.019719039, 0.013811458])
    bias = np.array([0, 0, 0, 0, 0, 0, 0])
    return gains, bias


def meta_xml(file):
    tree = ET.parse(file)
    time = tree.find("SatelliteInfo").find("CenterTime").find("Acamera").text
    saa = tree.find("SatelliteInfo").find("SolarAzimuth").text
    sza = tree.find("SatelliteInfo").find("SolarZenith").text
    roll = tree.find("SatelliteInfo").find("RollSatelliteAngle").text
    pitch = tree.find("SatelliteInfo").find("PitchSatelliteAngle").text
    yaw = tree.find("SatelliteInfo").find("YawSatelliteAngle").text
    return time, saa, sza, roll, pitch, yaw


def image_tif(file, blocksize=None):
    return GdalReadTifIterator(file, blocksize=blocksize)


def get(infile, blocksize=None):
    # outpath = os.path.dirname(infile)
    dirname = os.path.dirname(infile)
    basename = os.path.basename(infile)
    calib_file_path = glob.glob(dirname+os.sep+basename[0:49] + "*.calib.xml")[0]
    meta_file_path = glob.glob(dirname+os.sep+basename[0:49] + "*.meta.xml")[0]
    # calib_file_path=infile.replace(".tif", ".calib.xml")
    # meta_file_path = infile.replace(".tif", ".meta.xml")
    gains, bias = calib_xml(calib_file_path)
    time, saa_, sza_, roll, pitch, yaw = meta_xml(meta_file_path)
    year, month, day, hour, minute, second = \
        int(time[0:4]), int(time[5:7]), int(time[8:10]), int(time[11:13]), int(time[14:16]), int(time[17:19])
    data_Iterator = image_tif(infile, blocksize=blocksize)
    # 观测几何暂时不使用迭代器生成
    # sza = np.full_like(lon_grid, fill_value=sza_)
    # saa = np.full_like(lon_grid, fill_value=saa_)
    # vza = np.full_like(lon_grid, fill_value=5)
    # vaa = np.full_like(lon_grid, fill_value=-100)
    # 观测几何是一个临时策略，以后根据相关数据的发布进行调整
    sza = sza_
    saa = saa_
    vza = 5
    if basename[50] == "A":
        vaa = -100
    elif basename[50] == "B":
        vaa = 100

    num_443, num_490, num_520, num_555, num_670, nirs_num, nirl_num = 1, 2, None, 3, 4, 5, 6,
    nwvis = 4
    red = num_670

    return [sza, vza, saa, vaa, gains, bias, data_Iterator, year, month, day, num_443, num_490, num_520, num_555,
            num_670, nirs_num, nirl_num, nwvis, red]


if __name__ == '__main__':
    # file=r"E:\SDGSAT-1\KX10_MII_20220311_E114.52_N23.00_202200020689_L4A"+os.sep+"KX10_MII_20220311_E114.52_N23.00_202200020689_L4A.calib.xml"
    # calib_xml(file)

    metafile = r"E:\SDGSAT-1\KX10_MII_20220311_E114.52_N23.00_202200020689_L4A" + os.sep + "KX10_MII_20220311_E114.52_N23.00_202200020689_L4A.meta.xml"
    meta_xml(metafile)
