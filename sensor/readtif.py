# -*- coding: utf-8 -*-
"""
@author: li wenkai
@contact: lwk1542@hotmail.com
@software: pycharm anaconda python3.7 
@file: readtif.py
@time: 2022/1/10 16:48
@desc:
"""
import numpy as np
from osgeo import gdal, osr
import osgeo
import warnings
warnings.filterwarnings("ignore")




class GdalReadTifIterator(object):
    """
    2022.11.7
    这是一个迭代器，通过块状读取、块状处理，利于节约内存开支，以后高分辨率TIFF数据全部使用这个处理
    """

    def __init__(self, in_file, blocksize: int = None):
        """
        Args:
            in_file: 文件路径
            blocksize: 一次要处理的数据的行数，默认50
        """
        self.in_file = in_file  # Tiff文件
        self.blocksize = blocksize
        self.dataset = gdal.Open(self.in_file)
        self.XSize = self.dataset.RasterXSize  # 网格的X轴像素数量
        self.YSize = self.dataset.RasterYSize  # 网格的Y轴像素数量
        self.bands = self.dataset.RasterCount
        self.GeoTransform = self.dataset.GetGeoTransform()  # 投影转换信息
        self.ProjectionInfo = self.dataset.GetProjection()  # 投影信息
        # 投影转经纬度坐标
        self.prosrs = osr.SpatialReference()
        if int(osgeo.__version__[0]) >= 3:
            # GDAL 3 changes axis order: https://github.com/OSGeo/gdal/issues/1546
            self.prosrs.SetAxisMappingStrategy(osgeo.osr.OAMS_TRADITIONAL_GIS_ORDER)

        self.prosrs.ImportFromWkt(self.dataset.GetProjection())
        self.geosrs = self.prosrs.CloneGeogCS()

        self.iter_count = self.YSize // self.blocksize
        self.surplus_rows = self.YSize % self.blocksize
        if self.surplus_rows == 0:
            self.iter_count -= 1

        print(
            "imagery info:rows={0},columns={1},bands={2},block_rows={3},surplus_rows={4}".format(self.YSize, self.XSize,
                                                                                                 self.bands,
                                                                                                 self.blocksize,
                                                                                                 self.surplus_rows))
        # print(" this imagery will loop {0} times".format(self.iter_count))
        self.iter_num = 0
        self.lag = 0

    def __iter__(self):
        return self

    def __next__(self):
        # if self.iter_num <= self.iter_count-1:
        if self.iter_num <= self.iter_count:
            if (self.iter_num == self.iter_count) and (self.surplus_rows > 0):
                y_offset = self.surplus_rows
            else:
                y_offset = self.blocksize
            data = self.dataset.ReadAsArray(0, self.lag, self.XSize, y_offset)

            x_range = range(0, self.XSize)
            y_range = range(0, y_offset)
            x, y = np.meshgrid(x_range, y_range)
            lon_ = self.GeoTransform[0] + x * self.GeoTransform[1] + y * self.GeoTransform[2]
            lat_ = (self.GeoTransform[3] + self.iter_num * self.blocksize * self.GeoTransform[5]) + \
                   x * self.GeoTransform[4] + y * self.GeoTransform[5]
            ct = osr.CoordinateTransformation(self.prosrs, self.geosrs)
            coords = np.array(ct.TransformPoints(np.vstack([lon_.flatten(), lat_.flatten()]).T))
            lat, lon = coords[:, 1].reshape(lat_.shape), coords[:, 0].reshape(lat_.shape)
            self.iter_num += 1
            self.lag = self.iter_num * y_offset
            return data, lon, lat
        else:
            raise StopIteration()  # 表示至此停止迭代


if __name__ == '__main__':
    infiles = [
        r"E:\SDGSAT-1\test\KX10_MII_20220407_E115.85_N23.36_202200035633_L4A/KX10_MII_20220407_E115.85_N23.36_202200035633_L4A_A.tif",
        r"E:\SDGSAT-1\test\KX10_MII_20220407_E115.85_N23.36_202200035633_L4A/KX10_MII_20220407_E115.85_N23.36_202200035633_L4A_B.tif"]
    for i in GdalReadTifIterator(infiles[0]):
        data, lon, lat = i
        print(data.shape)

