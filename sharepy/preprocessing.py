# -*- coding: utf-8 -*-
"""
@Time    : 2023/1/2 21:57
@FileName: preprocessing.py
@Project : git_repository
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""
import numpy as np
from osgeo import gdal
from osgeo import osr
import os


def convert_coordinates(infile):
    """
    将投影转为WGS84经纬度
    """
    outfile = os.path.dirname(infile) + os.sep + os.path.splitext(os.path.basename(infile))[0] + "_WGS84.tif"
    gdal.Warp(outfile, infile, dstSRS="+proj=longlat +datum=WGS84 +no_defs")
    return outfile


def clip(infile, south, north, west, east):
    outfile = os.path.dirname(infile) + os.sep + os.path.splitext(os.path.basename(infile))[0] + "_studyArea.tif"
    src_ds = gdal.Open(infile)
    # bands_count=src_ds.RasterCount
    im_width = src_ds.RasterXSize  # 栅格矩阵的列数
    im_height = src_ds.RasterYSize  # 栅格矩阵的行数
    im_geotrans = src_ds.GetGeoTransform()  # 获取仿射矩阵信息
    # im_proj = src_ds.GetProjection()  # 获取投影信息
    xOrigin = im_geotrans[0]
    yOrigin = im_geotrans[3]
    pixelWidth = im_geotrans[1]
    pixelHeight = -im_geotrans[5]
    lon = xOrigin + pixelWidth * np.arange(0, im_width)
    lat = yOrigin + pixelHeight * np.arange(0, im_height)
    llon, llat = np.meshgrid(lon, lat)
    data = src_ds.ReadAsArray(0, 0, im_width, im_height)

    # 找到有效数据和要求范围的最小边界，用于裁剪数据
    mask = np.isnan(data).all(axis=0)   # 数据掩膜，某个像元位置的所有波段是否全为nan，如果全是nan，则为True
    llon[mask] = np.nan
    llat[mask] = np.nan
    south_img = np.nanmin(llat)
    north_img = np.nanmax(llat)
    south_ = np.nanmax([south_img, south])
    north_ = np.nanmin([north_img, north])
    llon[llat > north_] = np.nan
    llon[llat < south_] = np.nan
    west_img = np.nanmin(llon)
    east_img = np.nanmax(llon)
    west_ = np.nanmax([west_img, west])
    east_ = np.nanmin([east_img, east])
    if (south_>north_)|(west_>east_):
        print("no valid region...")
        return None

    # 裁剪
    ds = gdal.Translate(outfile, src_ds, projWin=[west_, north_, east_, south_])
    ds=None
    return outfile





