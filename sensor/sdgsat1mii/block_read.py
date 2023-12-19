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
import glob
import numpy as np
from osgeo import gdal, osr
import osgeo
from xml.etree import ElementTree as ET
import warnings
warnings.filterwarnings("ignore")


def meta_xml(file):
    tree = ET.parse(file)
    time = tree.find("SatelliteInfo").find("CenterTime").find("Acamera").text
    saa = tree.find("SatelliteInfo").find("SolarAzimuth").text
    sza = tree.find("SatelliteInfo").find("SolarZenith").text
    roll = tree.find("SatelliteInfo").find("RollSatelliteAngle").text
    pitch = tree.find("SatelliteInfo").find("PitchSatelliteAngle").text
    yaw = tree.find("SatelliteInfo").find("YawSatelliteAngle").text

    return time, float(saa), float(sza), float(roll), float(pitch), float(yaw)


def calib_xml(xmlfile):
    # xmlfile = "L4A.calib.xml"
    # f = open(file, "r", encoding="gb2312") # gb2312格式不好用
    # datasource = f.read()
    # per = ET.parse(datasource)
    new_caliCoeff = np.array([[0.051560133, 0], [0.036241353, 0], [0.023316835, 0], [0.015849666, 0], [0.016096381, 0],
                      [0.019719039, 0], [0.013811458, 0]])/10

    # 自己交叉定标后的系数
    # new_caliCoeff = [[0.00501, 1.49503],
    #                  [0.0031, 2.00318],
    #                  [0.00249, 0.13565],
    #                  [0.00178, -0.43103],
    #                  [0.00144, 0.13846],
    #                  [0.00197, -0.06615],
    #                  [0.00112, 0.01839]
    #                  ]
    # new_caliCoeff = [[0.00636, 0.0],
    #                  [0.00437, 0.0],
    #                  [0.00256, 0.0],
    #                  [0.00154, 0.0],
    #                  [0.0016, 0.0],
    #                  [0.00179, 0.0],
    #                  [0.00117, 0.0]]
    new_caliCoeff = [
        [0.00543, 0.99903],
        [0.00329, 1.65215],
        [0.00249, 0.05158],
        [0.00163, -0.26043],
        [0.0013, 0.18868],
        [0.00147, 0.05295],
        [0.00096, 0.04909]
    ]
    new_caliCoeff = [
        [0.00543, 0.0],
        [0.00329, 0.0],
        [0.00249, 0.0],
        [0.00163, 0.0],
        [0.0013, 0.0],
        [0.00147, 0.0],
        [0.00096, 0.0]
    ]

    new_caliCoeff = [[0.00579, 0.57391],
                     [0.00348, 1.31149],
                     [0.00255, -0.07411],
                     [0.0016, -0.20136],
                     [0.00127, 0.20652],
                     [0.00142, 0.06843],
                     [0.00091, 0.05786]]

    # new_caliCoeff = [[0.00503, 1.45823],
    #                  [0.00314, 1.95308],
    #                  [0.00249, 0.14742],
    #                  [0.00162, -0.06789],
    #                  [0.0012, 0.32672],
    #                  [0.00139, 0.07772],
    #                  [0.00092, 0.06119]]

    gains = np.array([i[0] for i in new_caliCoeff])
    bias = np.array([i[1] for i in new_caliCoeff])
    return gains, bias


class ReadIterator(object):
    """
    2022.11.7
    这是一个迭代器，通过块状读取、块状处理，利于节约内存开支，以后高分辨率TIFF数据全部使用这个处理
    """

    def __init__(self, in_file, blocksize: int = None, rows: int = None, columns: int = None):
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
            data = self.dataset.ReadAsArray(0, self.lag, self.XSize, y_offset)*1.
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
            self.iter_num += 1
            self.lag = self.iter_num * y_offset

            #
            dirname = os.path.dirname(self.in_file)
            basename = os.path.basename(self.in_file)
            name_id = "_".join(basename.split("_")[0:7])
            calib_file_path = glob.glob(dirname + os.sep + name_id + "*.calib.xml")[0]
            meta_file_path = glob.glob(dirname + os.sep + name_id + "*.meta.xml")[0]
            gains, offsets = calib_xml(calib_file_path)
            time, saa_, sza_, roll, pitch, yaw = meta_xml(meta_file_path)

            vaa = np.zeros_like(lat)+100
            vza = np.zeros_like(vaa) + 5
            saa = np.zeros_like(vaa) + saa_
            sza = np.zeros_like(vaa) + sza_

            return data.transpose(1, 2, 0), gains, offsets, lon, lat, vaa, vza, saa, sza

        else:
            raise StopIteration()  # 表示至此停止迭代


if __name__ == '__main__':
    infiles = [
        r"E:\SDGSAT-1\test\KX10_MII_20220407_E115.85_N23.36_202200035633_L4A/KX10_MII_20220407_E115.85_N23.36_202200035633_L4A_A.tif",
        r"E:\SDGSAT-1\test\KX10_MII_20220407_E115.85_N23.36_202200035633_L4A/KX10_MII_20220407_E115.85_N23.36_202200035633_L4A_B.tif"]
    for i in ReadIterator(infiles[0]):
        data, lon, lat = i
        print(data.shape)