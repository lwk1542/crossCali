# -*- coding: utf-8 -*-
"""
@Time    : 2023/10/8 11:24
@FileName: test.py
@Project : atmospheric_correction
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""
import os.path
from xml.etree import ElementTree as ET
import bs4
import lxml

file_dir = r"G:\SDGsat\calibration\sea\202303supply\reference\S3B_OL_1_EFR____20230219T200010_20230219T200310_20230220T090654_0179_076_199_2700_PS2_O_NT_003.SEN3"
file = os.path.join(file_dir, "xfdumanifest.xml")
soup = bs4.BeautifulSoup(open(file), 'lxml')
rows = int(soup.findAll('sentinel3:rows')[0].text)
columns = int(soup.findAll('sentinel3:columns')[0].text)

