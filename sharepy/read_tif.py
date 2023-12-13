# -*- coding: utf-8 -*-
"""
@Time    : 2023/4/3 11:02
@FileName: read_tif.py
@Project : git_repository
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""
from osgeo import gdal
import numpy as np


class GdalReadTif(object):
    def __init__(self, in_file):
        self.in_file = in_file  # Tiff或者ENVI文件
        dataset = gdal.Open(self.in_file)
        self.XSize = dataset.RasterXSize  # 网格的X轴像素数量
        self.YSize = dataset.RasterYSize  # 网格的Y轴像素数量
        self.GeoTransform = dataset.GetGeoTransform()  # 投影转换信息
        self.ProjectionInfo = dataset.GetProjection()  # 投影信息

    def get_data(self, band):
        """
        band: 读取第几个通道的数据
        """
        dataset = gdal.Open(self.in_file)
        band = dataset.GetRasterBand(band)
        data = band.ReadAsArray()
        return data

    def get_lon_lat(self):
        """
        获取经纬度信息
        """
        gtf = self.GeoTransform
        x_range = range(0, self.XSize)
        y_range = range(0, self.YSize)
        x, y = np.meshgrid(x_range, y_range)
        lon = gtf[0] + x * gtf[1] + y * gtf[2]
        lat = gtf[3] + x * gtf[4] + y * gtf[5]
        return lon, lat

    def get_info(self):
        return self.XSize, self.YSize