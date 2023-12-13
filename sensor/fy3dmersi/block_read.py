# -*- coding: utf-8 -*-
"""
@Time    : 2023/10/8 16:17
@FileName: block_read.py
@Project : atmospheric_correction
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""
import h5py
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")


class ReadIterator(object):
    """
    2022.11.7
    这是一个迭代器，通过块状读取、块状处理，利于节约内存开支
    fy-3d mersi
    """

    def __init__(self, in_file, blocksize: int = None, rows: int = None, columns: int = None):
        """
        Args:
            in_file: 文件路径
            blocksize: 一次要处理的数据的行数，默认50
        """
        self.in_file = in_file  # 文件
        self.blocksize = blocksize
        self.XSize = columns  # 网格的X轴像素数量
        self.YSize = rows  # 网格的Y轴像素数量
        # self.bands_name = [470, 550, 650, 865, 1380, 1640, 2130, 412, 443, 490, 555, 670, 709, 746, 866, 905, 936, 940,
        #                    1030] # 前面几个波段先不要
        self.bands_name = [412, 443, 490, 555, 670, 709, 746, 865, 905, 936, 940, 1030]
        # self.bands_name = [412, 443, 470, 490, 550, 555, 650, 670, 709, 746, 865, 866, 905, 936, 940, 1030]
        self.iter_count = self.YSize // self.blocksize
        self.surplus_rows = self.YSize % self.blocksize
        if self.surplus_rows == 0:
            self.iter_count -= 1
        print("分块数量={0}".format(self.iter_count))
        print("imagery info:rows={0},columns={1},bands={2},block_rows={3},"
              "surplus_rows={4}".format(self.YSize, self.XSize, self.bands_name.__len__(), self.blocksize,
                                        self.surplus_rows))
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
            size = self.bands_name.__len__()
            # data = np.zeros(shape=(y_offset, self.XSize, size)) + np.nan
            ds = h5py.File(self.in_file, mode="r")
            ob = ds["Data/EV_1KM_RefSB"]
            fillvalue = ob.attrs["FillValue"]
            data = ob[:, self.lag:self.lag + y_offset, :]*1.  # 15波段
            data[data == fillvalue] = np.nan
            data = data.transpose((1, 2, 0))[:, :, 3:]  # 可见光近红外

            # 查看官方文档 FY3D_MERSI_SRF_Pub-V2.1-201902.pdf
            cal = ds["Calibration/" + "VIS_Cal_Coeff"][7:, :]
            cal_0 = cal[:, 0].reshape(1, 1, -1)
            cal_1 = cal[:, 1].reshape(1, 1, -1)
            cal_2 = cal[:, 2].reshape(1, 1, -1)
            Ref = (data * data * cal_2 + data * cal_1 + cal_0)/100.
            # Ltoa= Ref * E0 / pi
            E0 = np.array(
                [1700.734, 1903.334, 1968.184, 1830.053, 1504.914, 1399.233, 1277.788, 955.2415, 884.8099, 828.4215,
                 820.4936, 680.8728]).reshape(1, 1, -1)
            E0 = ds.attrs["Solar_Irradiance"][7:].reshape(1, 1, -1)

            Ltoa = Ref * E0 / np.pi / 10.
            data = Ltoa

            # FY3D_MERSI_GBAL_L1_20231026_0730_1000M_MS.HDF,FY3D_MERSI_GBAL_L1_20231026_0730_GEO1K_MS.HDF
            _id = os.path.basename(self.in_file).split("_")
            _geofile = os.path.join(os.path.dirname(self.in_file), '_'.join(_id[0:6])+"_GEO1K_MS.HDF")
            geo_ds = h5py.File(_geofile, mode="r")

            latitude = _read_geo(ds=geo_ds, id="Geolocation/" + "Latitude", range=[self.lag, self.lag + y_offset])
            longitude = _read_geo(ds=geo_ds, id="Geolocation/" + "Longitude", range=[self.lag, self.lag + y_offset])
            sza = _read_geo(ds=geo_ds, id="Geolocation/" + "SolarZenith", range=[self.lag, self.lag + y_offset])
            saa = _read_geo(ds=geo_ds, id="Geolocation/" + "SolarAzimuth", range=[self.lag, self.lag + y_offset])
            vza = _read_geo(ds=geo_ds, id="Geolocation/" + "SensorZenith", range=[self.lag, self.lag + y_offset])
            vaa = _read_geo(ds=geo_ds, id="Geolocation/" + "SensorAzimuth", range=[self.lag, self.lag + y_offset])

            gains = np.ones_like(E0)
            offsets = np.zeros_like(E0)
            self.iter_num += 1
            self.lag = self.iter_num * y_offset
            return data, gains, offsets, longitude, latitude, vaa, vza, saa, sza
        else:
            raise StopIteration()  # 表示至此停止迭代


def _read_geo(ds: object, id: str, range: list[int, int]) -> np.ndarray:
    ob = ds[id]
    data = ob[range[0]:range[1], :] * 1.
    fillvalue = ob.attrs["FillValue"]
    data[data == fillvalue] = np.nan
    solpe = ob.attrs["Slope"]
    data = data * solpe
    intercept = ob.attrs["Intercept"]
    data = data + intercept
    return data


if __name__ == '__main__':
    infiles = [
        r"E:\SDGSAT-1\test\KX10_MII_20220407_E115.85_N23.36_202200035633_L4A/KX10_MII_20220407_E115.85_N23.36_202200035633_L4A_A.tif",
        r"E:\SDGSAT-1\test\KX10_MII_20220407_E115.85_N23.36_202200035633_L4A/KX10_MII_20220407_E115.85_N23.36_202200035633_L4A_B.tif"]
    for i in ReadIterator(infiles[0]):
        data, lon, lat = i
        print(data.shape)
