# -*- coding: utf-8 -*-
"""
@Time    : 2023/10/8 16:17
@FileName: block_read.py
@Project : atmospheric_correction
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""

import numpy as np
import warnings
from scipy import interpolate
import netCDF4 as nc
# netcdf4读取nc会自动乘以scale_factor,加add_offset
import os

warnings.filterwarnings("ignore")


def geophy():
    band_names = {
        400: 'Oa01_radiance', 412: 'Oa02_radiance',
        443: 'Oa03_radiance', 490: 'Oa04_radiance',
        510: 'Oa05_radiance', 560: 'Oa06_radiance',
        620: 'Oa07_radiance', 665: 'Oa08_radiance',
        674: 'Oa09_radiance', 681: 'Oa10_radiance',
        709: 'Oa11_radiance', 754: 'Oa12_radiance',
        760: 'Oa13_radiance', 764: 'Oa14_radiance',
        767: 'Oa15_radiance', 779: 'Oa16_radiance',
        865: 'Oa17_radiance', 885: 'Oa18_radiance',
        900: 'Oa19_radiance', 940: 'Oa20_radiance',
        1020: 'Oa21_radiance',
    }
    band_index = {
        400: 0, 412: 1, 443: 2, 490: 3,
        510: 4, 560: 5, 620: 6, 665: 7,
        674: 8, 681: 9, 709: 10, 754: 11,
        760: 12, 764: 13, 767: 14, 779: 15,
        865: 16, 885: 17, 900: 18, 940: 19,
        1020: 20}

    return band_names, band_index


class ReadIterator(object):
    """
    2022.11.7
    这是一个迭代器，通过块状读取、块状处理，利于节约内存开支
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
        self.bands_name = geophy()[0]
        self.bands_index = geophy()[1]
        self.iter_count = self.YSize // self.blocksize
        self.surplus_rows = self.YSize % self.blocksize
        if self.surplus_rows == 0:
            self.iter_count -= 1
        print("分块数量={0}".format(self.iter_count))
        print(
            "imagery info:rows={0},columns={1},bands={2},block_rows={3},surplus_rows={4}".format(self.YSize, self.XSize,
                                                                                                 self.bands_name.keys(),
                                                                                                 self.blocksize,
                                                                                                 self.surplus_rows))
        self.iter_num = 0
        self.lag = 0
        self.nc_datasets = {}
        for key in self.bands_name.keys():
            self.nc_datasets[key] = nc.Dataset(os.path.join(self.in_file, self.bands_name[key]) + ".nc", mode="r")[
                self.bands_name[key]]

        latlon = nc.Dataset(os.path.join(self.in_file, "geo_coordinates.nc"), mode="r")
        self.nc_lat = latlon["latitude"]
        self.nc_lon = latlon["longitude"]
        self.geometric = nc.Dataset(os.path.join(self.in_file, "tie_geometries.nc"), mode="r")

    def __iter__(self):
        return self

    def __next__(self):
        if self.iter_num <= self.iter_count:
            if (self.iter_num == self.iter_count) and (self.surplus_rows > 0):
                y_offset = self.surplus_rows
            else:
                y_offset = self.blocksize
            size = self.bands_name.__len__()
            data = np.zeros(shape=(y_offset, self.XSize, size)) + np.nan
            gains = np.zeros(shape=(size)) + np.nan
            offsets = np.zeros(shape=(size)) + np.nan
            for key in self.bands_name.keys():
                ds = self.nc_datasets[key]
                ip = self.bands_index[key]
                data[:, :, ip], gains[ip], offsets[ip] = read_nc(ds=ds, lag=self.lag, offset=y_offset)
                gains[ip], offsets[ip] = 1., 0.  # netcdf4已经自动处理，所以这里赋值1,0

            # lat,lon
            lat_, scale_, add_ = read_nc(ds=self.nc_lat, lag=self.lag, offset=y_offset)
            lat = lat_ #* scale_ + add_
            del scale_, add_
            lon_, scale_, add_ = read_nc(ds=self.nc_lon, lag=self.lag, offset=y_offset)
            lon = lon_ #* scale_ + add_
            del scale_, add_

            # sza, vza, saa, vaa
            geome = {}
            for band_ in ["OAA", "OZA", "SAA", "SZA"]:
                temp_, scale_, add_ = read_nc(ds=self.geometric[band_], lag=self.lag, offset=y_offset)
                # geome[band_] = tie_correc(ds=self.geometric, data_org=temp_*scale_+add_, shape_target=lon.shape)
                geome[band_] = tie_correc(ds=self.geometric, data_org=temp_, shape_target=lon.shape)
                del temp_, scale_, add_

            self.iter_num += 1
            self.lag = self.iter_num * y_offset
            return data, gains, offsets, lon, lat, geome["OAA"], geome["OZA"], geome["SAA"], geome["SZA"]
        else:
            raise StopIteration()  # 表示至此停止迭代


def read_nc(ds, lag, offset):
    value = ds[lag:lag + offset, :] * 1.
    try:
        add_offset = ds.getncattr("add_offset")
    except:
        add_offset = 0
    try:
        scale_factor = ds.getncattr("scale_factor")
    except:
        scale_factor = 1.
    Fillvalue = ds.getncattr("_FillValue")
    value[value == Fillvalue] = np.nan
    valid_max = ds.getncattr("valid_max")
    valid_min = ds.getncattr("valid_min")
    value[value < valid_min] = np.nan
    value[value > valid_max] = np.nan
    return value, scale_factor, add_offset


def tie_correc(ds, data_org, shape_target):
    # window for tiepoint data read
    ac = ds.getncattr('ac_subsampling_factor')
    al = ds.getncattr('al_subsampling_factor')

    rows, columns = data_org.shape
    # rows_org = np.arange(0, rows, 1)*al
    # rows_intp = np.arange(0, rows * al, 1)
    # column_org = np.arange(0, columns, 1) * ac
    # column_intp = np.arange(0, columns * ac, 1)

    column_org = np.linspace(0, shape_target[1], num=columns, endpoint=True)
    rows_org = np.linspace(0, shape_target[0], num=rows, endpoint=True)
    rows_intp = np.arange(0, shape_target[0], 1)
    column_intp = np.arange(0, shape_target[1], 1)

    x, y = np.meshgrid(column_org, rows_org)
    x_, y_ = np.meshgrid(column_intp, rows_intp)
    return interpolate.griddata((y.flatten(), x.flatten()), data_org.flatten(), (y_, x_), method='linear')


if __name__ == '__main__':
    infiles = [
        r"E:\SDGSAT-1\test\KX10_MII_20220407_E115.85_N23.36_202200035633_L4A/KX10_MII_20220407_E115.85_N23.36_202200035633_L4A_A.tif",
        r"E:\SDGSAT-1\test\KX10_MII_20220407_E115.85_N23.36_202200035633_L4A/KX10_MII_20220407_E115.85_N23.36_202200035633_L4A_B.tif"]
    for i in ReadIterator(infiles[0]):
        data, lon, lat = i
        print(data.shape)
