# -*- coding: utf-8 -*-
"""
@Time    : 2024/11/17 12:18
@FileName: test.py
@Project : git_repository
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""
from pyhdf.SD import SD, SDC

# 创建或打开 HDF4 文件
hdf_file = SD("example.hdf", SDC.WRITE | SDC.CREATE)

# 写入全局属性
att = hdf_file.attr('Relative Humidity')
att.set(SDC.FLOAT32, 30.0)


# 写入其他数据或属性（可选）
data = [[1, 2, 3], [4, 5, 6]]
sds = hdf_file.create("Dataset", SDC.FLOAT32, (2, 3))
sds[:] = data

# 确保关闭文件以保存更改
hdf_file.end()

print("Attribute 'Version' has been written.")