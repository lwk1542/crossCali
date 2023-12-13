# -*- coding: utf-8 -*-
"""
@Time    : 2023/4/3 14:35
@FileName: reproject.py
@Project : git_repository
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""
from osgeo import gdal,osr
import os
import osgeo


def convert_coordinates(infile):
    """
    将投影转为WGS84经纬度
    """
    outfile = os.path.dirname(infile) + os.sep + os.path.splitext(os.path.basename(infile))[0] + "_WGS84.tif"
    if os.path.exists(outfile):
        return outfile
    else:
        gdal.Warp(outfile, infile, dstSRS="+proj=longlat +datum=WGS84 +no_defs")
    return outfile


def lonlat2geo(dataset, lon, lat):
    '''
    将经纬度坐标转为投影坐标（具体的投影坐标系由给定数据确定）
    :param dataset: GDAL地理数据
    :param lon: 地理坐标lon经度
    :param lat: 地理坐标lat纬度
    :return: 经纬度坐标(lon, lat)对应的投影坐标
    '''
    target = osr.SpatialReference(wkt=dataset.GetProjection())
    source = osr.SpatialReference()
    source.ImportFromEPSG(4326)
    if int(osgeo.__version__[0]) >= 3:
        # GDAL 3 changes axis order: https://github.com/OSGeo/gdal/issues/1546
        source.SetAxisMappingStrategy(osgeo.osr.OAMS_TRADITIONAL_GIS_ORDER)
        target.SetAxisMappingStrategy(osgeo.osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(source, target)
    coords = transform.TransformPoint(lon, lat)
    return coords[:2]