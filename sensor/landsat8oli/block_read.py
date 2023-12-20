# -*- coding: utf-8 -*-
"""
@Time    : 2023/10/25 17:33
@FileName: block_read.py
@Project : atmospheric_correction
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""
import os
import numpy as np
from osgeo import gdal, osr
import osgeo
import warnings
from . import landsat_metadata
warnings.filterwarnings("ignore")


class ReadIterator(object):
    """
    2022.11.7
    这是一个迭代器，通过块状读取、块状处理，利于节约内存开支，以后高分辨率TIFF数据全部使用这个处理
    """

    def __init__(self, in_file, blocksize: int, rows: int, columns: int):
        """
        Args:
            in_file: 文件路径
            blocksize: 一次要处理的数据的行数，例如150
        """
        self.in_file = in_file  # 文件夹
        mtl_file = self.in_file + os.sep + os.path.basename(self.in_file) + "_MTL.xml"
        self.blocksize = blocksize
        self.bands = 6

        tif_file = self.in_file + os.sep + os.path.basename(self.in_file) + "_B1.TIF"
        self.dataset = gdal.Open(tif_file)
        self.XSize = self.dataset.RasterXSize  # 网格的X轴像素数量
        self.YSize = self.dataset.RasterYSize  # 网格的Y轴像素数量
        self.GeoTransform = self.dataset.GetGeoTransform()  # 投影转换信息
        self.ProjectionInfo = self.dataset.GetProjection()

        datetime_obj, self.gains,  self.offsets,  self.rows,  self.columns = landsat_metadata.mtl(file=mtl_file)
        datetime = datetime_obj.strftime("%Y-%m-%dT%H:%M:%S.%f")  # 2023-04-10T15:39:26.2045389Z
        self.year, self.month, self.day, self.hour, self.minute, self.second = \
            datetime[0:4], datetime[5:7], datetime[8:10], datetime[12:14], datetime[16:18], datetime[20:]

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

        band_list = ["B1.TIF", "B2.TIF", "B3.TIF", "B4.TIF", "B5.TIF", "B6.TIF", "B7.TIF"]
        self.files = []
        for i, band in enumerate(band_list):
            self.files.append(mtl_file.replace("MTL.xml", band))

        band_list2 = ["VAA.TIF", "VZA.TIF", "SAA.TIF", "SZA.TIF"]
        self.geofiles = []
        for i, band in enumerate(band_list2):
            self.geofiles.append(mtl_file.replace("MTL.xml", band))

        print("imagery info:rows={0},columns={1},bands={2},block_rows={3},surplus_rows={4}".
              format(self.YSize, self.XSize, self.bands, self.blocksize, self.surplus_rows))
        # print(" this imagery will loop {0} times".format(self.iter_count))
        self.iter_num = 0
        self.lag = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.iter_num <= self.iter_count:
            if (self.iter_num == self.iter_count) and (self.surplus_rows > 0):
                y_offset = self.surplus_rows
            else:
                y_offset = self.blocksize
            for i, file in enumerate(self.files):
                ds = gdal.Open(file)
                _ = ds.ReadAsArray(0, self.lag, self.XSize, y_offset)*1.
                if i == 0:
                    data = _
                else:
                    data = np.dstack([data, _])
            data[data == 0] = np.nan
            x_range = range(0, self.XSize)
            y_range = range(0, y_offset)
            x, y = np.meshgrid(x_range, y_range)
            lon_ = self.GeoTransform[0] + x * self.GeoTransform[1] + y * self.GeoTransform[2]
            lat_ = (self.GeoTransform[3] + self.iter_num * self.blocksize * self.GeoTransform[5]) + \
                   x * self.GeoTransform[4] + y * self.GeoTransform[5]
            ct = osr.CoordinateTransformation(self.prosrs, self.geosrs)
            coords = np.array(ct.TransformPoints(np.vstack([lon_.flatten(), lat_.flatten()]).T))
            lat, lon = coords[:, 1].reshape(lat_.shape), coords[:, 0].reshape(lat_.shape)
            vaa = gdal.Open(self.geofiles[0]).ReadAsArray(0, self.lag, self.XSize, y_offset) * .01
            vza = gdal.Open(self.geofiles[1]).ReadAsArray(0, self.lag, self.XSize, y_offset) * .01
            saa = gdal.Open(self.geofiles[2]).ReadAsArray(0, self.lag, self.XSize, y_offset) * .01
            sza = gdal.Open(self.geofiles[3]).ReadAsArray(0, self.lag, self.XSize, y_offset) * .01
            self.iter_num += 1
            self.lag = self.iter_num * y_offset
            return data, np.array(self.gains), np.array(self.offsets), lon, lat, vaa, vza, saa, sza
        else:
            raise StopIteration()  # 表示至此停止迭代


# if __name__ == '__main__':
#     infiles = [
#         r"E:\SDGSAT-1\test\KX10_MII_20220407_E115.85_N23.36_202200035633_L4A/KX10_MII_20220407_E115.85_N23.36_202200035633_L4A_A.tif",
#         r"E:\SDGSAT-1\test\KX10_MII_20220407_E115.85_N23.36_202200035633_L4A/KX10_MII_20220407_E115.85_N23.36_202200035633_L4A_B.tif"]
#     for i in ReadIterator(infiles[0]):
#         data, lon, lat = i
#         print(data.shape)