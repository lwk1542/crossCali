# -*- coding: utf-8 -*-
"""
@Time    : 2023/12/27 15:34
@FileName: test.py
@Project : atmospheric_correction
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""
from xml.etree import ElementTree as ET
file= r"G:\SDGsat\calibration\sea\2023\calibration\process\target\KX10_MII_20230124_W62.25_N17.66_202300012093_L4B/KX10_MII_20230124_W62.25_N17.66_202300012093_L4B.calib.xml"
tree = ET.parse(file)
root = tree.getroot()
PROCESSING_RECORD = root.findall('IMAGE_ATTRIBUTES')[0]
dto_string = PROCESSING_RECORD.find('DATE_ACQUIRED').text +"T"+PROCESSING_RECORD.find('SCENE_CENTER_TIME').text
dto_string = dto_string.split('.')[0]+"."+dto_string.split('.')[1][:6]  #  strptime 函数在处理微秒时有长度限制，无法直接解析超过6位数的微秒部分
# datetime_obj = datetime.datetime.strptime(dto_string, "%Y-%m-%dT%H:%M:%S.%f")
RESCALING = root.findall("LEVEL1_RADIOMETRIC_RESCALING")[0]
gains = []
offsets = []
for i in range(7):
    gains.append(float(RESCALING.find("RADIANCE_MULT_BAND_"+str(i+1)).text))
    offsets.append(float(RESCALING.find("RADIANCE_ADD_BAND_" + str(i + 1)).text))