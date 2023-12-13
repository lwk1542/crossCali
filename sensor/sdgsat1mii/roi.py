# -*- coding: utf-8 -*-
"""
@Time    : 2023/9/21 8:39
@FileName: roi.py
@Project : atmospheric_correction
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""
from osgeo import gdal, osr
import os
import osgeo


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


def roi(filepath: str,
        rangelonlat: list[float, float, float, float] = [117., 41., 126., 28.],
        resize: float = 0.03) -> str:
    outtif = filepath + os.sep + os.path.basename(filepath) + "_ROI.tif"
    if os.path.exists(outtif):
        return outtif
        # if os.path.getsize(outtif)/float(1024)/float(1024) < 50:
        # os.remove(outtif)
    if not os.path.exists(outtif):
        print("merge and clip..........")
        filesep = [filepath + os.sep + os.path.basename(filepath)+"_A.tif",
                   filepath + os.sep + os.path.basename(filepath)+"_B.tif"]
        outds = gdal.BuildVRT("", filesep, separate=False)
        dataset = gdal.Open(filesep[0])
        coords1 = lonlat2geo(dataset, rangelonlat[0], rangelonlat[1])
        coords2 = lonlat2geo(dataset, rangelonlat[2], rangelonlat[3])
        spatialwin = [coords1[0], coords1[1], coords2[0], coords2[1]]
        options_list = [
            '-outsize ' + str(resize * 100) + '% ' + str(resize * 100) + '%'
        ]
        options_string = " ".join(options_list)
        outtif_temp = os.path.splitext(outtif)[0]+"_temp.tif"
        outds_ = gdal.Translate(outtif_temp, outds, projWin=spatialwin)
        outds_1 = gdal.Translate(outtif, outds_, options=options_string)
        del outds_, outds, outds_1
        os.remove(outtif_temp)
    return outtif
