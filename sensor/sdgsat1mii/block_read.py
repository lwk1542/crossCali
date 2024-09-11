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
from lxml import etree


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
    root = etree.parse(xmlfile)
    or_gains = np.array(
        [float(root.find("./RADIOMETRIC_CALIBRATION/MII/VERSION/RADIANCE_GAIN_BAND_"+str(i+1)).text) for i in range(7)])/10
    or_bias = np.array(
        [float(root.find("./RADIOMETRIC_CALIBRATION/MII/VERSION/RADIANCE_BIAS_BAND_" + str(i + 1)).text) for i in range(7)])

    new_caliCoeff = [[1.10337, -0.25476],
                     [1.10715, -0.13425],
                     [1.35326, -0.08069],
                     [0.93589, - 0.23609],
                     [0.95152, - 0.04716],
                     [1.14712, - 0.26284],
                     [1.36286, - 0.27304]]
    new_caliCoeff = [[1.10337, -0.40476],
                     [1.10715, -0.00425],
                     [1.35326, 2.18069],
                     [0.85589, - 0.23609],
                     [0.95152, - 0.01716],
                     [1.14712, - 0.26284],
                     [1.36286, - 0.27304]]

    new_caliCoeff = [[1.26254, -0.91464],
                     [1.05123, -0.0297],
                     [0.8127, 0.75015],
                     [0.68587, 0.5454],
                     [0.58692, 0.35871],
                     [0.50841, 0.12226],
                     [0.47304, 0.08019]]

    new_caliCoeff = [[1.0, 0.0],
                     [1.0, 0.0],
                     [1.0, 0.0],
                     [1.0, 0.0],
                     [1.0, 0.0],
                     [1.0, 0.0],
                     [1.0, 0.0]]

    gains = np.array([i[0] for i in new_caliCoeff])
    bias = np.array([i[1] for i in new_caliCoeff])
    return or_gains, or_bias, gains, bias


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

            dirname = os.path.dirname(self.in_file)
            basename = os.path.basename(self.in_file)
            name_id = "_".join(basename.split("_")[0:7])
            calib_file_path = glob.glob(dirname + os.sep + name_id + "*.calib.xml")[0]
            meta_file_path = glob.glob(dirname + os.sep + name_id + "*.meta.xml")[0]
            or_gains, or_bias, gains, offsets = calib_xml(calib_file_path)
            time, saa_, sza_, roll, pitch, yaw = meta_xml(meta_file_path)

            vaa = np.zeros_like(lat)+100
            vza = np.zeros_like(vaa) + 5
            saa = np.zeros_like(vaa) + saa_
            sza = np.zeros_like(vaa) + sza_

            return data.transpose(1, 2, 0) * or_gains.reshape(1, 1, -1) + or_bias.reshape(1, 1, -1), \
                   gains, offsets, lon, lat, vaa, vza, saa, sza

        else:
            raise StopIteration()  # 表示至此停止迭代


if __name__ == '__main__':
    infiles = [
        r"E:\SDGSAT-1\test\KX10_MII_20220407_E115.85_N23.36_202200035633_L4A/KX10_MII_20220407_E115.85_N23.36_202200035633_L4A_A.tif",
        r"E:\SDGSAT-1\test\KX10_MII_20220407_E115.85_N23.36_202200035633_L4A/KX10_MII_20220407_E115.85_N23.36_202200035633_L4A_B.tif"]
    for i in ReadIterator(infiles[0]):
        data, lon, lat = i
        print(data.shape)