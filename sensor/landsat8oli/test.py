# -*- coding: utf-8 -*-
"""
@Time    : 2023/12/19 20:11
@FileName: test.py
@Project : atmospheric_correction
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""
import landsat_metadata
from xml.etree import ElementTree as ET
import datetime
from osgeo import gdal

def mtl(file:str):
    tree = ET.parse(file)
    root = tree.getroot()
    PROCESSING_RECORD = root.findall('IMAGE_ATTRIBUTES')[0]
    dto_string = PROCESSING_RECORD.find('DATE_ACQUIRED').text + PROCESSING_RECORD.find('SCENE_CENTER_TIME').text
    datetime_obj = datetime.datetime.strptime(dto_string, "%Y-%m-%d%H:%M:%S.%f")
    RESCALING = root.findall("LEVEL1_RADIOMETRIC_RESCALING")
    gains = []
    offsets = []
    for i in range(7):
        gains.append(float(RESCALING.find("RADIANCE_MULT_BAND_"+str(i+1)).text))
        offsets.append(float(RESCALING.find("RADIANCE_ADD_BAND_" + str(i + 1)).text))
    PROJECTION_ATTRIBUTES = root.findall("PROJECTION_ATTRIBUTES")
    rows = PROJECTION_ATTRIBUTES.find("REFLECTIVE_LINES")
    columns = PROJECTION_ATTRIBUTES.find("REFLECTIVE_SAMPLES")
    return datetime_obj, gains, offsets, rows, columns


def img():
    file = r"G:\SDGsat\calibration\sea\2023\insitu\imagery\forAeronetOC\landsat\LC08_L1TP_014032_20230410_20230420_02/LC08_L1TP_014032_20230410_20230420_02_T1_B1.TIF"
    ds = gdal.Open(file)
    GeoTransform = ds.GetGeoTransform()  # 投影转换信息
    ProjectionInfo = ds.GetProjection()
    pass

if __name__ == '__main__':
    mtl_file = r"G:\SDGsat\calibration\sea\2023\insitu\imagery\forAeronetOC\landsat\LC08_L1TP_014032_20230410_20230420_02_T1\LC08_L1TP_014032_20230410_20230420_02_T1_MTL.xml"
    mtl(file=mtl_file)
    # img()