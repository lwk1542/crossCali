# -*- coding: utf-8 -*-
"""
@Time    : 2023/10/25 17:45
@FileName: read_img2.py
@Project : atmospheric_correction
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""
import os
from . import block_read
import warnings
# import bs4
import glob
warnings.filterwarnings("ignore")

from xml.etree import ElementTree as ET
from osgeo import gdal

def meta_xml(file):
    tree = ET.parse(file)
    time = tree.find("SatelliteInfo").find("CenterTime").find("Acamera").text
    # saa = tree.find("SatelliteInfo").find("SolarAzimuth").text
    # sza = tree.find("SatelliteInfo").find("SolarZenith").text
    # roll = tree.find("SatelliteInfo").find("RollSatelliteAngle").text
    # pitch = tree.find("SatelliteInfo").find("PitchSatelliteAngle").text
    # yaw = tree.find("SatelliteInfo").find("YawSatelliteAngle").text
    # columns = tree.find("ImageInfo").find("NumPixels").text
    # rows = tree.find("ImageInfo").find("NumLines").text
    year, month, day, hour, minute, second = \
        int(time[0:4]), int(time[5:7]), int(time[8:10]), int(time[11:13]), int(time[14:16]), int(float(time[17:]))
    return year, month, day, hour, minute#, columns, rows


def get(infile, blocksize=None):

    dirname = os.path.dirname(infile)
    basename = os.path.basename(infile)
    name_id = "_".join(basename.split("_")[0:7])
    meta_file_path = glob.glob(dirname + os.sep + name_id + "*.meta.xml")[0]
    year, month, day, hour, minute = meta_xml(meta_file_path)

    dataset = gdal.Open(infile)
    columns = dataset.RasterXSize  # 网格的X轴像素数量
    rows = dataset.RasterYSize  # 网格的Y轴像素数量

    data_Iterator = block_read.ReadIterator(infile, blocksize=blocksize, rows=rows, columns=columns)
    num_443, num_490, num_520, num_555, num_670, nirs_num, nirl_num = 1, 2, None, 3, 4, 5, 6,
    nwvis = 4
    red = num_670
    return [data_Iterator, year, month, day, num_443, num_490, num_520, num_555,
            num_670, nirs_num, nirl_num, nwvis, red, rows, columns]

