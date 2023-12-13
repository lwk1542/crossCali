# -*- coding: utf-8 -*-
"""
@author: li wenkai
@contact: lwk1542@hotmail.com
@software: pycharm anaconda python3.7 
@file: readfile_cc.py
@time: 2021/1/25 14:35
@desc: 逐步将数据读取迁移到base_info()类中
"""
import glob
import itertools
import os

import h5py
import numpy as np
from netCDF4 import Dataset

from sharepy import predefine


def preprodata(f, dataID):
    # 读取数据，处理其填充值、最大、最小、增益和偏移系数
    data = f[dataID][()]
    if '_FillValue' in f[dataID].attrs.keys():
        data[data == f[dataID].attrs['_FillValue']] = np.nan
    if 'valid_min' in f[dataID].attrs.keys():
        data[data < f[dataID].attrs['valid_min']] = np.nan
    if 'valid_max' in f[dataID].attrs.keys():
        data[data > f[dataID].attrs['valid_max']] = np.nan
    if 'scale_factor' in f[dataID].attrs.keys():
        data = data * f[dataID].attrs['scale_factor']
    if 'add_offset' in f[dataID].attrs.keys():
        data = data + f[dataID].attrs['add_offset']
    return data


def lut_select(infile: str = "H1A_RICH_OCT_L1B_20020910T011452_20020910T012818_1671_10.H5"):
    """

    Args:
        infile ():

    Returns: 根据文件名确定大气校正相关的查找表lut路径

    """
    # 根据文件名确定查找表路径
    # sensorID = os.path.basename(infile)[0:3]
    # print('sensor ID: ' + sensorID)
    # if any([sensorID == 'H1A', sensorID == 'H1B', sensorID == 'H1C'])
    if all(idxi in os.path.basename(infile) for idxi in ['H1A', 'OCT']):
        lut_path = r'LUT/HY1A_COCTS_LUTs'
    elif all(idxi in os.path.basename(infile) for idxi in ['H1B', 'OCT']):
        lut_path = r'LUT/HY1B_COCTS_LUTs'
    elif all(idxi in os.path.basename(infile) for idxi in ['H1C', 'OCT']):
        lut_path = r'LUT/HY1C_COCTS_LUTs'
    elif all(idxi in os.path.basename(infile) for idxi in ['H1C', 'czi']):
        lut_path = r'LUT/HY1C_CZI_LUTs'
    elif all(idxi in os.path.basename(infile) for idxi in ['H1C', 'uvi']):
        lut_path = r'LUT/HY1C_UVI_LUTs'
    elif all(idxi in os.path.basename(infile) for idxi in ['MOD']):
        lut_path = r'LUT/Terra_modis_LUTs'
    elif all(idxi in os.path.basename(infile) for idxi in ['MYD']):
        lut_path = r'LUT/Aqua_modis_LUTs'
    else:
        print('unidentified satellite sensor')
        return None
    return lut_path


def read_sensorinfo(sensorinfo_file: str = "msl12_sensor_info.dat"):
    """

    Args:
        sensorinfo_file (): msl12_sensor_info.dat文件，有标准格式

    Returns:
        传感器部分参数
    """
    # f = open(r"D:\share\modis\aqua/msl12_sensor_info.dat")
    f = open(sensorinfo_file)
    lines = f.readlines()
    # filter note
    lines = [i for i in lines if not "#" in i.lower()]
    wavelength = [float(i.split("= ", 1)[-1].split()[0]) for i in lines if "lambda(" in i.lower()]
    F0 = [float(i.split("= ", 1)[-1].split()[0]) for i in lines if "f0(" in i.lower()]
    Tau_r = [float(i.split("= ", 1)[-1].split()[0]) for i in lines if "tau_r(" in i.lower()]
    k_oz = [float(i.split("= ", 1)[-1].split()[0]) for i in lines if "k_oz(" in i.lower()]
    t_co2 = [float(i.split("= ", 1)[-1].split()[0]) for i in lines if "t_co2(" in i.lower()]
    k_no2 = [float(i.split("= ", 1)[-1].split()[0]) for i in lines if "k_no2(" in i.lower()]
    a_h2o = [float(i.split("= ", 1)[-1].split()[0]) for i in lines if "a_h2o(" in i.lower()]
    b_h2o = [float(i.split("= ", 1)[-1].split()[0]) for i in lines if "b_h2o(" in i.lower()]
    c_h2o = [float(i.split("= ", 1)[-1].split()[0]) for i in lines if "c_h2o(" in i.lower()]
    d_h2o = [float(i.split("= ", 1)[-1].split()[0]) for i in lines if "d_h2o(" in i.lower()]
    e_h2o = [float(i.split("= ", 1)[-1].split()[0]) for i in lines if "e_h2o(" in i.lower()]
    f_h2o = [float(i.split("= ", 1)[-1].split()[0]) for i in lines if "f_h2o(" in i.lower()]
    g_h2o = [float(i.split("= ", 1)[-1].split()[0]) for i in lines if "g_h2o(" in i.lower()]
    awhite = [float(i.split("= ", 1)[-1].split()[0]) for i in lines if "awhite(" in i.lower()]
    aw = [float(i.split("= ", 1)[-1].split()[0]) for i in lines if "aw(" in i.lower()]
    bbw = [float(i.split("= ", 1)[-1].split()[0]) for i in lines if "bbw(" in i.lower()]
    oobwv = [float(i.split("= ", 1)[-1].split()[0]) for i in lines if "oobwv" in i.lower()]
    ooblw = [float(i.split("= ", 1)[-1].split()[0]) for i in lines if "ooblw" in i.lower()]
    wed = [float(i.split("= ", 1)[-1].split()[0]) for i in lines if "wed(" in i.lower()]
    waph = [float(i.split("= ", 1)[-1].split()[0]) for i in lines if "waph(" in i.lower()]

    return [wavelength, F0, Tau_r, k_oz, t_co2, k_no2, a_h2o, b_h2o, c_h2o, d_h2o, e_h2o, f_h2o, g_h2o, awhite, aw, bbw,
            oobwv, ooblw, wed, waph]


def hy1abc_l1ab(infile=None, north: float = None, south: float = None, west: float = None, east: float = None):
    """
    Args:
        infile (): hy1系列卫星L1A/L1B文件
    Returns:
        图像数据、观测几何、经纬度,大气层顶辐照度F0，各气体消光参数等
    """
    # infile = r'G:\high_quality\test/H1B_RICH_OCT_L1A_20130502T004830_20130502T01061131609_10.h5'

    bands = np.array([412, 443, 490, 520, 565, 670, 750, 865])
    if all(idxi in os.path.basename(infile) for idxi in ['H1A', 'OCT']):
        Fo = np.array([171.50607089839818, 169.91507744342238, 204.22006711632858, 192.92464799828767, 186.325213894711,
                       154.8971134548406, 132.64710158528882, 100.59576039607016])
        kno2 = np.array([5.971E-19, 4.966E-19, 2.786E-19, 1.819E-19, 7.200E-20, 7.805E-21, 1.176E-21, 5.205E-23])
        koz = np.array([0.780E-03, 3.342E-03, 2.288E-02, 4.967E-02, 1.174E-01, 4.443E-02, 9.477E-03, 2.191E-03])
        taur = np.array([3.129E-01, 2.337E-01, 1.554E-01, 1.235E-01, 8.535E-02, 4.325E-02, 2.752E-02, 1.576E-02])
        tco2 = np.array([1., 1., 1., 1., 1., 1., 1., 1.])
        a_h2o = np.array([1., 1., 1., 1., 1., 1., 1., 1.])
        b_h2o = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
        c_h2o = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
        d_h2o = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
        e_h2o = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
        f_h2o = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
        g_h2o = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
        # Band-averaged water absorption
        aw = np.array([5.200E-02, 7.500E-02, 1.630E-02, 3.900E-02, 6.990E-02, 4.496E-01, 2.678E+00, 4.569E+00])

        # Band-averaged water backscatter
        bbw = np.array([3.300E-03, 2.400E-03, 1.600E-03, 1.200E-03, 8.463E-04, 4.153E-04, 2.586E-04, 1.441E-04])

    elif all(idxi in os.path.basename(infile) for idxi in ['H1B', 'OCT']):
        Fo = np.array(
            [173.38355174632883, 191.17458512445108, 197.73106054804956, 183.49599438680198, 180.43210560151945,
             149.63902485662874, 126.7797882993995, 95.01985515574087])
        koz = np.array([0.780E-03, 3.342E-03, 2.288E-02, 4.967E-02, 1.174E-01, 4.443E-02, 9.477E-03, 2.191E-03])
        kno2 = np.array([5.971E-19, 4.966E-19, 2.786E-19, 1.819E-19, 7.200E-20, 7.805E-21, 1.176E-21, 5.205E-23])
        taur = np.array([3.129E-01, 2.337E-01, 1.554E-01, 1.235E-01, 8.535E-02, 4.325E-02, 2.752E-02, 1.576E-02])
        tco2 = np.array([1., 1., 1., 1., 1., 1., 1., 1.])
        a_h2o = np.array([1., 1., 1., 1., 1., 1., 1., 1.])
        b_h2o = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
        c_h2o = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
        d_h2o = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
        e_h2o = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
        f_h2o = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
        g_h2o = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
        # Band-averaged water absorption
        aw = np.array([5.200E-02, 7.500E-02, 1.630E-02, 3.900E-02, 6.990E-02, 4.496E-01, 2.678E+00, 4.569E+00])
        # Band-averaged water backscatter
        bbw = np.array([3.300E-03, 2.400E-03, 1.600E-03, 1.200E-03, 8.463E-04, 4.153E-04, 2.586E-04, 1.441E-04])

    elif all(idxi in os.path.basename(infile) for idxi in ['H1C', 'OCT']):
        Fo = np.array(
            [171.71656646582093, 188.34941558108426, 197.10349197278518, 184.03588330937973, 179.78116454798246,
             150.40907258492732, 126.8182416769412, 96.25892906742042])
        # 瑞利光学厚度
        taur = np.array([0.3129, 0.2337, 0.1554, 0.1235, 0.08535, 0.04325, 0.02752, 0.01576])
        kno2 = np.array([5.964517378098426e-19, 5.01667526749566e-19, 2.7848945048908554e-19, 1.802352368086221e-19,
                         7.184451458522253e-20, 7.79749419156795e-21, 1.1646586908876065e-21, 5.0782980858941893e-23])
        koz = np.array([0.780E-03, 3.342E-03, 2.288E-02, 4.967E-02, 1.174E-01, 4.443E-02, 9.477E-03, 2.191E-03])
        tco2 = np.array([1., 1., 1., 1., 1., 1., 1., 1.])
        a_h2o = np.array([1., 1., 1., 1., 1., 1., 1., 1.])
        b_h2o = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
        c_h2o = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
        d_h2o = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
        e_h2o = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
        f_h2o = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
        g_h2o = np.array([0., 0., 0., 0., 0., 0., 0., 0.])

        # Band-averaged water absorption
        aw = np.array([5.200E-02, 7.500E-02, 1.630E-02, 3.900E-02, 6.990E-02, 4.496E-01, 2.678E+00, 4.569E+00])
        # Band-averaged water backscatter
        bbw = np.array([3.300E-03, 2.400E-03, 1.600E-03, 1.200E-03, 8.463E-04, 4.153E-04, 2.586E-04, 1.441E-04])

        # 消光截面cm2粒子数
        # koz = np.array([5.383974200705576e-23, 1.6887334980611554e-22, 9.286214356410431e-22, 1.931216823979986e-21,
        #                 4.494106553398089e-21, 1.769899601233302e-21, 3.9318364922493383e-22, 8.995093366407616e-23])

    elif all(idxi in os.path.basename(infile) for idxi in ['H1C', 'uvi']):
        Fo = np.array([193.29278985414103, 180.88234968907233, 156.2625743580722, 1108.69541092487526])
        taur = np.array([0.59, 0.42])
        # 待计算
        kno2 = np.array([])
        koz = np.array([])
        tco2 = np.array([1., 1., 1., 1.])
        a_h2o = np.array([1., 1., 1., 1.])
        b_h2o = np.array([0., 0., 0., 0.])
        c_h2o = np.array([0., 0., 0., 0.])
        d_h2o = np.array([0., 0., 0., 0.])
        e_h2o = np.array([0., 0., 0., 0.])
        f_h2o = np.array([0., 0., 0., 0.])
        g_h2o = np.array([0., 0., 0., 0.])
        bands = np.array([460, 560, 650, 825])
        # Band-averaged water absorption
        aw = np.array([5.200E-02, 7.500E-02, 1.630E-02, 3.900E-02])
        # Band-averaged water backscatter
        bbw = np.array([3.300E-03, 2.400E-03, 1.600E-03, 1.200E-03])
    elif all(idxi in os.path.basename(infile) for idxi in ['H1C', 'czi']):
        Fo = np.array([193.29278985414103, 180.88234968907233, 156.2625743580722, 1108.69541092487526])
        taur = np.array([0.2, 0.094, 0.05, 0.021])
        # 待计算
        kno2 = np.array([])
        koz = np.array([])
        tco2 = np.array([1., 1., 1., 1.])
        a_h2o = np.array([1., 1., 1., 1.])
        b_h2o = np.array([0., 0., 0., 0.])
        c_h2o = np.array([0., 0., 0., 0.])
        d_h2o = np.array([0., 0., 0., 0.])
        e_h2o = np.array([0., 0., 0., 0.])
        f_h2o = np.array([0., 0., 0., 0.])
        g_h2o = np.array([0., 0., 0., 0.])
        bands = np.array([460, 560, 650, 825])
        # Band-averaged water absorption
        aw = np.array([5.200E-02, 7.500E-02, 1.630E-02, 3.900E-02])
        # Band-averaged water backscatter
        bbw = np.array([3.300E-03, 2.400E-03, 1.600E-03, 1.200E-03])

    else:
        Fo = np.array(['no sensor'])
        taur = np.array(['no sensor'])
        kno2 = np.array(['no sensor'])
        koz = np.array(['no sensor'])
        tco2 = np.array(['no sensor'])
        a_h2o = np.array(['no sensor'])
        b_h2o = np.array(['no sensor'])
        c_h2o = np.array(['no sensor'])
        d_h2o = np.array(['no sensor'])
        e_h2o = np.array(['no sensor'])
        f_h2o = np.array(['no sensor'])
        g_h2o = np.array(['no sensor'])
        bands = np.array(['no sensor'])
        print('no expected HY-1 file, executing next file')

    if 'L1B' in os.path.basename(infile):
        bands_str = ['L_' + str(i) for i in bands]
    elif 'L1A' in os.path.basename(infile):
        bands_str = ['DN_' + str(i) for i in bands]
    else:
        bands_str = ['no sensor']
        print('no expected product level L1A or L1B, executing next file')

    h5f = h5py.File(infile, 'r')
    try:
        scales = h5f['/Calibration/Calibration Coefficients Scale factor'][()]
    except:
        scales = np.array([1., 1., 1., 1., 1., 1., 1., 1.])
    try:
        offsets = h5f['/Calibration/Calibration Coefficients Offsets factor'][()]
    except:
        offsets = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
    try:
        gains = h5f['/Calibration/Vicarious Calibration gain factor'][()]
        # 官方发布的H1C的替代定标系数是错的，零时全部改为1
        gains = np.array([1., 1., 1., 1., 1., 1., 1., 1.])
    except:
        gains = np.array([1., 1., 1., 1., 1., 1., 1., 1.])

    navi = h5f['/Navigation Data']
    lon = preprodata(navi, 'Longitude')
    lat = preprodata(navi, 'Latitude')
    # lon = h5f['/Navigation Data/Longitude'][()]
    # lat = h5f['/Navigation Data/Latitude'][()]
    if north is None:
        north = np.max(lat)
    if south is None:
        south = np.min(lat)
    if west is None:
        west = np.min(lon)
    if east is None:
        east = np.max(lon)
    loc = np.where((south < lat) & (lat < north) & (west < lon) & (lon < east))
    if loc[0].size < predefine.thresholds().pixels_num:  # 不到500各像元直接不计算了
        return ()
    up, low, left, right = np.min(loc[0]), np.max(loc[0]), np.min(loc[1]), np.max(loc[1])

    lat = lat[up:low, left:right]
    lon = lon[up:low, left:right]

    vaa = h5f['/Navigation Data/Satellite Azimuth Angle'][()][up:low, left:right]
    vza = h5f['/Navigation Data/Satellite Zenith Angle'][()][up:low, left:right]
    try:
        sza = h5f['/Navigation Data/Solar Zenith Angle'][()][up:low, left:right]
        saa = h5f['/Navigation Data/Solar Azimuth Angle'][()][up:low, left:right]
    except:
        sza = h5f['/Navigation Data/Sun Zenith Angle'][()][up:low, left:right]
        saa = h5f['/Navigation Data/Sun Azimuth Angle'][()][up:low, left:right]

    # lt_ib = np.empty(shape=(lon.shape[0], lon.shape[1], bands.__len__()))
    dn = np.empty(shape=(lon.shape[0], lon.shape[1], bands.__len__()))
    print("calibration coefficients:")
    print("scale:")
    print(scales)
    print("scale:")
    print(offsets)
    print("Vicarious Calibration")
    print(gains)
    for i in range(bands.__len__()):
        dn[:, :, i] = h5f['/Geophysical Data/' + bands_str[i]][()][up:low, left:right]
        # if 'L1A' in os.path.basename(infile):
        dn[:, :, i] = (dn[:, :, i] * scales[i] + offsets[i]) * gains[i]  # 定标
    Zia_tabel = [a_h2o, b_h2o, c_h2o, d_h2o, e_h2o, f_h2o, g_h2o]
    out = (sza, vza, saa, vaa, lat, lon, dn, bands, Fo, taur, koz, kno2, tco2, Zia_tabel, aw, bbw)
    return out


def modis_l1b(infile: str = r'D:\HYproject\backup/MOD021KM.A2003105.0220.061.2017193203705.hdf',
              north: float = None, south: float = None, west: float = None, east: float = None):
    geofile = glob.glob(
        os.path.dirname(infile) + os.sep + os.path.basename(infile)[0:3] + '03.' + os.path.basename(infile)[
                                                                                   9:27] + '*.hdf')
    if geofile.__len__() < 1:
        return
    else:
        geofile = geofile[0]
    try:
        f = Dataset(infile, mode='r')
        f03 = Dataset(geofile, mode='r')
    except:
        print('cannot read MODIS file')
        return
    lat = f03.variables['Latitude'][()]
    lon = f03.variables['Longitude'][()]
    if north is None:
        north = np.max(lat)
    if south is None:
        south = np.min(lat)
    if west is None:
        west = np.min(lon)
    if east is None:
        east = np.max(lon)

    loc = np.where((south < lat) & (lat < north) & (west < lon) & (lon < east))
    if loc[0].size < predefine.thresholds().pixels_num:  # 不到500各像元直接不计算了
        return None

    up, low, left, right = np.min(loc[0]), np.max(loc[0]), np.min(loc[1]), np.max(loc[1])

    lt250 = f.variables['EV_250_Aggr1km_RefSB']
    scales = lt250.radiance_scales
    offset = lt250.radiance_offsets
    valid_min, valid_max = lt250.valid_range
    value250 = lt250[()] * 1.
    value250[(value250 < valid_min) | (valid_max < value250)] = np.nan
    scales = scales.reshape(-1, 1, 1)
    offset = offset.reshape(-1, 1, 1)
    value250 = (value250 - offset) * scales
    value250 = value250[:, up:low, left:right]

    lt500 = f.variables['EV_500_Aggr1km_RefSB']
    scales = lt500.radiance_scales
    offset = lt500.radiance_offsets
    valid_min, valid_max = lt500.valid_range
    value500 = lt500[()] * 1.
    value500[(value500 < valid_min) | (valid_max < value500)] = np.nan
    scales = scales.reshape(-1, 1, 1)
    offset = offset.reshape(-1, 1, 1)
    value500 = (value500 - offset) * scales
    value500 = value500[:, up:low, left:right]

    lt1000 = f.variables['EV_1KM_RefSB']
    scales = lt1000.radiance_scales
    offset = lt1000.radiance_offsets
    valid_min, valid_max = lt1000.valid_range
    value1000 = lt1000[()] * 1.
    value1000[(value1000 < valid_min) | (valid_max < value1000)] = np.nan
    scales = scales.reshape(-1, 1, 1)
    offset = offset.reshape(-1, 1, 1)
    value1000 = (value1000 - offset) * scales
    value1000 = value1000[:, up:low, left:right]

    Lt = np.zeros(shape=(low - up, right - left, 16))

    Lt[:, :, 0] = value1000[0, :, :]  # 412
    Lt[:, :, 1] = value1000[1, :, :]  # 444
    Lt[:, :, 2] = value500[0, :, :]  # 469
    Lt[:, :, 3] = value1000[2, :, :]  # 488
    Lt[:, :, 4] = value1000[3, :, :]  # 530
    Lt[:, :, 5] = value1000[4, :, :]  # 547
    Lt[:, :, 5] = value1000[4, :, :]  # 551
    Lt[:, :, 6] = value500[1, :, :]  # 554
    Lt[:, :, 7] = value250[0, :, :]  # 645
    Lt[:, :, 8] = value1000[5, :, :]  # 666
    Lt[:, :, 9] = value1000[7, :, :]  # 678
    Lt[:, :, 10] = value1000[9, :, :]  # 747
    Lt[:, :, 11] = value250[1, :, :]  # 857
    Lt[:, :, 12] = value1000[10, :, :]  # 867
    Lt[:, :, 13] = value500[2, :, :]  # 1242
    Lt[:, :, 14] = value500[3, :, :]  # 1628
    Lt[:, :, 15] = value500[4, :, :]  # 2113

    sza = f03.variables['SolarZenith'][()][up:low, left:right]
    saa = f03.variables['SolarAzimuth'][()][up:low, left:right]
    vza = f03.variables['SensorZenith'][()][up:low, left:right]
    vaa = f03.variables['SensorAzimuth'][()][up:low, left:right]
    lat = lat[up:low, left:right]
    lon = lon[up:low, left:right]

    if 'MOD' in os.path.basename(infile):
        bands = np.array([412, 443, 469, 488, 531, 551, 555, 645, 667, 678, 748, 859, 869, 1240, 1640, 2130])
        Fo = np.array(
            [172.632, 187.484, 205.878, 195.117, 185.699, 186.475, 183.869, 157.811, 151.694, 147.470, 127.873, 97.174,
             95.816, 45.467, 23.977, 9.885])
        taur = np.array(
            [3.161E-01, 2.369E-01, 1.914E-01, 1.603E-01, 1.130E-01, 9.936E-02, 9.432E-02, 5.082E-02, 4.431E-02,
             4.139E-02, 2.842E-02, 1.613E-02, 1.545E-02, 3.617E-03, 1.219E-03, 4.286E-04])
        # ozone吸收系数，单位nm,cm-1
        koz = np.array(
            [8.772E-04, 3.123E-03, 8.745E-03, 2.020E-02, 6.776E-02, 8.581E-02, 9.553E-02, 7.382E-02, 4.877E-02,
             3.803E-02, 1.212E-02, 2.347E-03, 1.990E-03, 0.000E+00, 0.000E+00, 0.000E+00])
        # koz=np.array([4.8261354990307185e-23, 1.4687267256743471e-22, 3.743039292885627e-22, 7.923124456786432e-22,
        #      2.6257544467439693e-21, 3.3426887281669384e-21, 3.7163229895487735e-21, 2.817242447338633e-21,
        #      1.87191607761548e-21, 1.4512921673811617e-21, 4.707701329513615e-22, 9.212897119373293e-23,
        #      7.450854052399815e-23, 0.0, 0.0, 0.0])
        kno2 = np.array(
            [5.915E-19, 4.993E-19, 3.938E-19, 2.910E-19, 1.547E-19, 1.208E-19, 9.445E-20, 1.382E-20, 6.965E-21,
             8.519E-21, 2.115E-21, 6.212E-23, 1.233E-22, 0.000E+00, 0.000E+00, 0.000E+00])
        tco2 = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.99941, 0.98896, 0.96965])
        a_h2o = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 9.9995e-01, 1.0000e-00, 9.9937e-01])
        b_h2o = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -3.3142e-03, -9.8591e-04, -1.4018e-02])
        c_h2o = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.2570e-04, 7.3884e-06, 9.1894e-04])
        d_h2o = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -5.6632e-06, -8.6648e-08, -4.8570e-05])
        e_h2o = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.6045e-07, 7.0068e-10, 1.4696e-06])
        f_h2o = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -2.3923e-09, 3.6264e-15, -2.2665e-08])
        g_h2o = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.4280e-11, -3.5516e-14, 1.3802e-10])
        Zia_tabel = [a_h2o, b_h2o, c_h2o, d_h2o, e_h2o, f_h2o, g_h2o]
    elif 'MYD' in os.path.basename(infile):
        bands = np.array([412, 443, 469, 488, 531, 547, 551, 555, 645, 667, 678, 748, 859, 869, 1240, 1640, 2130])
        Fo = np.array(
            [172.912, 187.622, 205.878, 194.933, 185.747, 186.539, 183.869, 157.811, 152.255, 148.052, 128.065, 97.174,
             95.824, 45.467, 23.977, 9.885])
        taur = np.array(
            [3.099E-01, 2.367E-01, 1.914E-01, 1.592E-01, 1.126E-01, 9.906E-02, 9.432E-02, 5.082E-02, 4.443E-02,
             4.146E-02, 2.849E-02, 1.613E-02, 1.540E-02, 3.617E-03, 1.219E-03, 4.286E-04])
        # koz=[1.987E-03,3.189E-03,8.745E-03,2.032E-02,6.838E-02,8.622E-02,9.553E-02,7.382E-02,4.890E-02,3.787E-02,1.235E-02,2.347E-03,1.936E-03,0.000E+00,0.000E+00,0.000E+00]
        koz = np.array([8.75007698770743e-23, 1.49391440215291e-22, 3.743039292885627e-22, 7.962263449842514e-22,
                        2.6523717256358085e-21, 3.3589517285129204e-21, 3.7163229895487735e-21, 2.817242447338633e-21,
                        1.884027318852435e-21, 1.452569444785557e-21, 4.782747475341676e-22, 9.212897119373293e-23,
                        7.187665910358705e-23, 0.0, 0.0, 0.0])

        kno2 = np.array(
            [5.814E-19, 4.985E-19, 3.938E-19, 2.878E-19, 1.525E-19, 1.194E-19, 9.445E-20, 1.382E-20, 7.065E-21,
             8.304E-21, 2.157E-21, 6.212E-23, 7.872E-23, 0.000E+00, 0.000E+00, 0.000E+00])
        tco2 = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.99941, 0.98896, 0.96965])

        # 水汽使用的时Terra的参数
        a_h2o = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 9.9995e-01, 1.0000e-00, 9.9937e-01])
        b_h2o = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -3.3142e-03, -9.8591e-04, -1.4018e-02])
        c_h2o = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.2570e-04, 7.3884e-06, 9.1894e-04])
        d_h2o = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -5.6632e-06, -8.6648e-08, -4.8570e-05])
        e_h2o = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.6045e-07, 7.0068e-10, 1.4696e-06])
        f_h2o = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -2.3923e-09, 3.6264e-15, -2.2665e-08])
        g_h2o = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.4280e-11, -3.5516e-14, 1.3802e-10])

    else:
        Fo = np.array(['no sensor'])
        taur = np.array(['no sensor'])
        kno2 = np.array(['no sensor'])
        koz = np.array(['no sensor'])
        tco2 = np.array(['no sensor'])
        a_h2o = np.array(['no sensor'])
        b_h2o = np.array(['no sensor'])
        c_h2o = np.array(['no sensor'])
        d_h2o = np.array(['no sensor'])
        e_h2o = np.array(['no sensor'])
        f_h2o = np.array(['no sensor'])
        g_h2o = np.array(['no sensor'])
        bands = np.array(['no sensor'])

    Lt = Lt / 10
    Zia_tabel = [a_h2o, b_h2o, c_h2o, d_h2o, e_h2o, f_h2o, g_h2o]
    out = (sza.data, vza.data, saa.data, vaa.data, lat.data, lon.data, Lt, bands, Fo, taur, koz, kno2, tco2, Zia_tabel)
    return out


def read_aerosol_info(infile=None):
    # 自己处理的文件
    f = h5py.File(infile, mode='r')
    aer_model_max = f["/geophysical_data/aer_model_max"][()]
    aer_model_min = f["/geophysical_data/aer_model_min"][()]
    aer_model_ratio = f["/geophysical_data/aer_model_ratio"][()]
    La_nirl = f["/geophysical_data/La_869"][()]
    bands = np.array([412, 443, 469, 488, 531, 547, 555, 645, 667, 678, 748, 859, 869])  # , 1240, 1640, 2130
    F0 = np.array([172.912, 187.622, 205.878, 194.933, 185.747, 186.539, 183.869, 157.811, 152.255, 148.052, 128.065,
                   97.174, 95.824])  # , 45.467, 23.977, 9.885
    Rrs = np.full(shape=(La_nirl.shape[0], La_nirl.shape[1], bands.__len__()), fill_value=np.nan)
    for i, band in enumerate(bands):
        # valid_max=f["/geophysical_data/Rrs_"+str(band)]
        Rrs[:, :, i] = f["/geophysical_data/Rrs_" + str(band)][()]

    # for i, band in enumerate(bands):
    #     taua[:,:,i]=f["/aerosol_information/aot_"+str(band)][()]

    lat = f["/navigation_data/latitude"][()]
    lon = f["/navigation_data/longitude"][()]
    sza = f["/geophysical_data/solz"][()]
    vza = f["/geophysical_data/senz"][()]
    saa = f["/geophysical_data/sola"][()]
    vaa = f["/geophysical_data/sena"][()]
    f.close()
    sensorID = os.path.basename(infile)[0:3]
    if sensorID == 'MOD':
        nirl_num = 12
    elif sensorID == 'MYD':
        Fo_nirl = 95.824
    else:
        Fo_nirl = 95.824
        print('no Fo in these sensor')

    return Rrs, F0, aer_model_ratio, La_nirl, F0[nirl_num], lat, lon, sza, vza, saa, vaa, aer_model_max, aer_model_min


def read_aerosol_info_seadas(infile=None):
    # seadas处理出来的文件
    f = h5py.File(infile, mode='r')
    aer_model_max = f["/geophysical_data/aer_model_max"][()] * 1. - 1
    aer_model_max[(aer_model_max < 0) | (aer_model_max > 79)] = np.nan
    aer_model_min = f["/geophysical_data/aer_model_min"][()] * 1. - 1
    aer_model_min[(aer_model_min < 0) | (aer_model_min > 79)] = np.nan
    aer_model_ratio = f["/geophysical_data/aer_model_ratio"][()]
    aer_model_ratio[(aer_model_ratio < 0) | (aer_model_ratio > 1)] = np.nan
    La_nirl = f["/geophysical_data/La_869"][()]
    La_nirl[(La_nirl < 0) | (La_nirl > 10)] = np.nan

    bands = np.array([412, 443, 469, 488, 531, 547, 555, 645, 667, 678, 748, 859, 869])  # , 1240, 1640, 2130
    F0 = np.array([172.912, 187.622, 205.878, 194.933, 185.747, 186.539, 183.869, 157.811, 152.255, 148.052, 128.065,
                   97.174, 95.824])  # , 45.467, 23.977, 9.885
    Rrs = np.full(shape=(La_nirl.shape[0], La_nirl.shape[1], bands.__len__()), fill_value=np.nan)
    for i, band in enumerate(bands):
        vari = f["/geophysical_data/Rrs_" + str(band)]
        offset = vari.attrs['add_offset'][0]
        scale = vari.attrs['scale_factor'][0]
        valid_max = vari.attrs['valid_max'][0]
        valid_min = vari.attrs['valid_min'][0]
        value = vari[()] * 1.
        value[(value < valid_min) | (value > valid_max)] = np.nan
        Rrs[:, :, i] = value * scale + offset

    # for i, band in enumerate(bands):
    #     taua[:,:,i]=f["/aerosol_information/aot_"+str(band)][()]

    lat = f["/navigation_data/latitude"][()]
    lat[(lat < -90) | (lat > 90)] = np.nan
    lon = f["/navigation_data/longitude"][()]
    lon[(lon < -180) | (lon > 180)] = np.nan

    vari = f["/geophysical_data/solz"]
    offset = vari.attrs['add_offset'][0]
    scale = vari.attrs['scale_factor'][0]
    valid_max = vari.attrs['valid_max'][0]
    valid_min = vari.attrs['valid_min'][0]
    value = vari[()] * 1.
    value[(value < valid_min) | (value > valid_max)] = np.nan
    sza = value * scale + offset

    vari = f["/geophysical_data/senz"]
    offset = vari.attrs['add_offset'][0]
    scale = vari.attrs['scale_factor'][0]
    valid_max = vari.attrs['valid_max'][0]
    valid_min = vari.attrs['valid_min'][0]
    value = vari[()] * 1.
    value[(value < valid_min) | (value > valid_max)] = np.nan
    vza = value * scale + offset

    vari = f["/geophysical_data/sola"]
    offset = vari.attrs['add_offset'][0]
    scale = vari.attrs['scale_factor'][0]
    valid_max = vari.attrs['valid_max'][0]
    valid_min = vari.attrs['valid_min'][0]
    value = vari[()] * 1.
    value[(value < valid_min) | (value > valid_max)] = np.nan
    saa = value * scale + offset

    vari = f["/geophysical_data/sena"]
    offset = vari.attrs['add_offset'][0]
    scale = vari.attrs['scale_factor'][0]
    valid_max = vari.attrs['valid_max'][0]
    valid_min = vari.attrs['valid_min'][0]
    value = vari[()] * 1.
    value[(value < valid_min) | (value > valid_max)] = np.nan
    vaa = value * scale + offset
    f.close()
    sensorID = os.path.basename(infile)[0:3]
    if sensorID == 'MOD':
        nirl_num = 12
    elif sensorID == 'MYD':
        Fo_nirl = 95.824
    else:
        Fo_nirl = 95.824
        print('no Fo in these sensor')

    return Rrs, F0, aer_model_ratio, La_nirl, F0[nirl_num], lat, lon, sza, vza, saa, vaa, aer_model_max, aer_model_min


def simulated_radiance_hy1(infile=None, north: float = None, south: float = None, west: float = None,
                           east: float = None):
    """
    Args:
        infile (): hy1系列卫星L1A/L1B文件
    Returns:
        图像数据、观测几何、经纬度,大气层顶辐照度F0，各气体消光参数等
    """
    # infile = r'G:\high_quality\test/H1B_RICH_OCT_L1A_20130502T004830_20130502T01061131609_10.h5'

    bands = np.array([412, 443, 490, 520, 565, 670, 750, 865])
    if all(idxi in os.path.basename(infile) for idxi in ['H1A', 'OCT']):
        Fo = np.array([171.50607089839818, 169.91507744342238, 204.22006711632858, 192.92464799828767, 186.325213894711,
                       154.8971134548406, 132.64710158528882, 100.59576039607016])
        kno2 = np.array([5.971E-19, 4.966E-19, 2.786E-19, 1.819E-19, 7.200E-20, 7.805E-21, 1.176E-21, 5.205E-23])
        koz = np.array([0.780E-03, 3.342E-03, 2.288E-02, 4.967E-02, 1.174E-01, 4.443E-02, 9.477E-03, 2.191E-03])
        taur = np.array([3.129E-01, 2.337E-01, 1.554E-01, 1.235E-01, 8.535E-02, 4.325E-02, 2.752E-02, 1.576E-02])
        tco2 = np.array([1., 1., 1., 1., 1., 1., 1., 1.])
        a_h2o = np.array([1., 1., 1., 1., 1., 1., 1., 1.])
        b_h2o = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
        c_h2o = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
        d_h2o = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
        e_h2o = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
        f_h2o = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
        g_h2o = np.array([0., 0., 0., 0., 0., 0., 0., 0.])

    elif all(idxi in os.path.basename(infile) for idxi in ['H1B', 'OCT']):
        Fo = np.array(
            [173.38355174632883, 191.17458512445108, 197.73106054804956, 183.49599438680198, 180.43210560151945,
             149.63902485662874, 126.7797882993995, 95.01985515574087])
        koz = np.array([0.780E-03, 3.342E-03, 2.288E-02, 4.967E-02, 1.174E-01, 4.443E-02, 9.477E-03, 2.191E-03])
        kno2 = np.array([5.971E-19, 4.966E-19, 2.786E-19, 1.819E-19, 7.200E-20, 7.805E-21, 1.176E-21, 5.205E-23])
        taur = np.array([3.129E-01, 2.337E-01, 1.554E-01, 1.235E-01, 8.535E-02, 4.325E-02, 2.752E-02, 1.576E-02])
        tco2 = np.array([1., 1., 1., 1., 1., 1., 1., 1.])
        a_h2o = np.array([1., 1., 1., 1., 1., 1., 1., 1.])
        b_h2o = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
        c_h2o = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
        d_h2o = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
        e_h2o = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
        f_h2o = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
        g_h2o = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
        # Band-averaged water absorption
        aw = np.array([5.200E-02, 7.500E-02, 1.630E-02, 3.900E-02, 6.990E-02, 4.496E-01, 2.678E+00, 4.569E+00])
        # Band-averaged water backscatter
        bbw = np.array([3.300E-03, 2.400E-03, 1.600E-03, 1.200E-03, 8.463E-04, 4.153E-04, 2.586E-04, 1.441E-04])

    elif all(idxi in os.path.basename(infile) for idxi in ['H1C', 'OCT']):
        Fo = np.array(
            [171.71656646582093, 188.34941558108426, 197.10349197278518, 184.03588330937973, 179.78116454798246,
             150.40907258492732, 126.8182416769412, 96.25892906742042])
        # 瑞利光学厚度
        taur = np.array([0.3129, 0.2337, 0.1554, 0.1235, 0.08535, 0.04325, 0.02752, 0.01576])
        kno2 = np.array([5.964517378098426e-19, 5.01667526749566e-19, 2.7848945048908554e-19, 1.802352368086221e-19,
                         7.184451458522253e-20, 7.79749419156795e-21, 1.1646586908876065e-21, 5.0782980858941893e-23])
        koz = np.array([0.780E-03, 3.342E-03, 2.288E-02, 4.967E-02, 1.174E-01, 4.443E-02, 9.477E-03])
        tco2 = np.array([1., 1., 1., 1., 1., 1., 1., 1.])
        a_h2o = np.array([1., 1., 1., 1., 1., 1., 1., 1.])
        b_h2o = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
        c_h2o = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
        d_h2o = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
        e_h2o = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
        f_h2o = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
        g_h2o = np.array([0., 0., 0., 0., 0., 0., 0., 0.])

        # Band-averaged water absorption
        aw = np.array([5.200E-02, 7.500E-02, 1.630E-02, 3.900E-02, 6.990E-02, 4.496E-01, 2.678E+00, 4.569E+00])
        # Band-averaged water backscatter
        bbw = np.array([3.300E-03, 2.400E-03, 1.600E-03, 1.200E-03, 8.463E-04, 4.153E-04, 2.586E-04, 1.441E-04])

        # 消光截面cm2粒子数
        # koz = np.array([5.383974200705576e-23, 1.6887334980611554e-22, 9.286214356410431e-22, 1.931216823979986e-21,
        #                 4.494106553398089e-21, 1.769899601233302e-21, 3.9318364922493383e-22, 8.995093366407616e-23])

    elif all(idxi in os.path.basename(infile) for idxi in ['H1C', 'czi']):
        Fo = np.array([193.29278985414103, 180.88234968907233, 156.2625743580722, 1108.69541092487526])
        taur = np.array([0.2, 0.094, 0.05, 0.021])
        # 待计算
        kno2 = np.array([])
        koz = np.array([])
        tco2 = np.array([1., 1., 1., 1.])
        a_h2o = np.array([1., 1., 1., 1.])
        b_h2o = np.array([0., 0., 0., 0.])
        c_h2o = np.array([0., 0., 0., 0.])
        d_h2o = np.array([0., 0., 0., 0.])
        e_h2o = np.array([0., 0., 0., 0.])
        f_h2o = np.array([0., 0., 0., 0.])
        g_h2o = np.array([0., 0., 0., 0.])
        bands = np.array([460, 560, 650, 825])
    else:
        Fo = np.array(['no sensor'])
        taur = np.array(['no sensor'])
        kno2 = np.array(['no sensor'])
        koz = np.array(['no sensor'])
        tco2 = np.array(['no sensor'])
        a_h2o = np.array(['no sensor'])
        b_h2o = np.array(['no sensor'])
        c_h2o = np.array(['no sensor'])
        d_h2o = np.array(['no sensor'])
        e_h2o = np.array(['no sensor'])
        f_h2o = np.array(['no sensor'])
        g_h2o = np.array(['no sensor'])
        bands = np.array(['no sensor'])
        print('no expected HY-1 file, executing next file')

    if 'crossCalibration' in os.path.basename(infile):
        bands_str = ['Lt_simu_' + str(i) for i in bands]
    else:
        bands_str = ['no sensor']
        print('no expected product level L1A or L1B, executing next file')

    h5f = h5py.File(infile, 'r')

    lon = h5f['/Navigation Data/lon'][()]
    lat = h5f['/Navigation Data/lat'][()]
    if north is None:
        north = np.max(lat)
    if south is None:
        south = np.min(lat)
    if west is None:
        west = np.min(lon)
    if east is None:
        east = np.max(lon)
    loc = np.where((south < lat) & (lat < north) & (west < lon) & (lon < east))
    if loc[0].size < predefine.thresholds().pixels_num:  # 不到500各像元直接不计算了
        return ()
    up, low, left, right = np.min(loc[0]), np.max(loc[0]), np.min(loc[1]), np.max(loc[1])

    lat = lat[up:low, left:right]
    lon = lon[up:low, left:right]

    vaa = h5f['/Geophysical Data/vaa_targetsensor'][()][up:low, left:right]
    vza = h5f['/Geophysical Data/vza_targetsensor'][()][up:low, left:right]

    sza = h5f['/Geophysical Data/sza_targetsensor'][()][up:low, left:right]
    saa = h5f['/Geophysical Data/saa_targetsensor'][()][up:low, left:right]

    dn = np.empty(shape=(lon.shape[0], lon.shape[1], bands.__len__()))
    for i in range(bands.__len__()):
        dn[:, :, i] = h5f['/Geophysical Data/' + bands_str[i]][()][up:low, left:right]

    Zia_tabel = [a_h2o, b_h2o, c_h2o, d_h2o, e_h2o, f_h2o, g_h2o]
    out = (sza, vza, saa, vaa, lat, lon, dn, bands, Fo, taur, koz, kno2, tco2, Zia_tabel, aw, bbw)
    return out


class base_info(object):
    def __init__(self, infile=None, sensorID=None, lut_path=None, mode="ac", north: float = None, south: float = None,
                 west: float = None, east: float = None):
        self.mode = mode
        self.sensorID = sensorID
        self.infile = infile
        self.lut_path = lut_path
        self.sensor_infos = None
        self.north = north
        self.south = south
        self.east = east
        self.west = west

    def run_main(self):
        # 1查找表
        if self.lut_path is None:
            self.lut_path = self.get_lookup_table()
        look_up_table = (self.lut_path + os.sep + "rayleigh", self.lut_path + os.sep + "aerosol")
        # 2传感器信息
        sensor_info = self.get_sensor_info(self.lut_path)
        # 3影像信息
        if self.mode == "ac":
            image_info = self.get_image_info()
        else:
            image_info = None
        return [look_up_table, sensor_info, image_info]

    def get_lookup_table(self):
        if self.sensorID is None:
            if all(idxi in os.path.basename(self.infile) for idxi in ['H1A', 'OCT']):
                lut_path = r'../share/cocts/hy1a'
            elif all(idxi in os.path.basename(self.infile) for idxi in ['H1B', 'OCT']):
                lut_path = r'../share/cocts/hy1b'
            elif all(idxi in os.path.basename(self.infile) for idxi in ['H1C', 'OCT']):
                lut_path = r'../share/cocts/hy1c'
            elif all(idxi in os.path.basename(self.infile) for idxi in ['H1C', 'czi']):
                lut_path = r'../share/czi/hy1c'
            elif all(idxi in os.path.basename(self.infile) for idxi in ['H1C', 'uvi']):
                lut_path = r'../share/uvi/hy1c'
            elif all(idxi in os.path.basename(self.infile) for idxi in ['MOD']):
                lut_path = r'../share/modis/terra'
            elif all(idxi in os.path.basename(self.infile) for idxi in ['MYD']):
                lut_path = r'../share/modis/aqua'
            elif all(idxi in os.path.basename(self.infile) for idxi in ['LC08']):
                lut_path = r'share/modis/oli'
            else:
                print('unidentified satellite sensor')
                lut_path = "look up table: unidentified satellite sensor"
        else:
            # 根据指定的传感器获取查找表路径
            print('sensorID: ' + self.sensorID)
            if self.sensorID == 'hy1acocts':
                lut_path = r"share" + os.sep + "cocts" + os.sep + "hy1a"
            elif self.sensorID == 'hy1bcocts':
                lut_path = r"share" + os.sep + "cocts" + os.sep + "hy1b"
            elif self.sensorID == 'hy1ccocts':
                lut_path = r"share" + os.sep + "cocts" + os.sep + "hy1c"
            elif self.sensorID == 'lc08oli':
                lut_path = r"share" + os.sep + "oli"
            elif self.sensorID == 'modist':
                lut_path = r"share" + os.sep + "modis"+os.sep+"terra"
            elif self.sensorID == 'modisa':
                lut_path = r"share" + os.sep + "modis"+os.sep+"aqua"

            else:
                lut_path = ""

        return lut_path

    def get_sensor_info(self, lut_path):
        """

        Args:
            sensorinfo_file (): msl12_sensor_info.dat文件，有标准格式

        Returns:
            传感器部分参数
        """
        # f = open(r"D:\share\modis\aqua/msl12_sensor_info.dat")
        sensorinfo_file = lut_path + os.sep + "msl12_sensor_info.dat"
        f = open(sensorinfo_file)
        lines = f.readlines()
        # filter note
        # filter note
        lines = [i for i in lines if not "#" in i.lower()]
        self.wavelength = [float(i.split("= ", 1)[-1].split()[0]) for i in lines if "lambda(" in i.lower()][0:8]
        F0 = [float(i.split("= ", 1)[-1].split()[0]) for i in lines if "f0(" in i.lower()][0:8]
        Tau_r = [float(i.split("= ", 1)[-1].split()[0]) for i in lines if "tau_r(" in i.lower()][0:8]
        k_oz = [float(i.split("= ", 1)[-1].split()[0]) for i in lines if "k_oz(" in i.lower()][0:8]
        t_co2 = [float(i.split("= ", 1)[-1].split()[0]) for i in lines if "t_co2(" in i.lower()][0:8]
        k_no2 = [float(i.split("= ", 1)[-1].split()[0]) for i in lines if "k_no2(" in i.lower()][0:8]
        a_h2o = [float(i.split("= ", 1)[-1].split()[0]) for i in lines if "a_h2o(" in i.lower()][0:8]
        b_h2o = [float(i.split("= ", 1)[-1].split()[0]) for i in lines if "b_h2o(" in i.lower()][0:8]
        c_h2o = [float(i.split("= ", 1)[-1].split()[0]) for i in lines if "c_h2o(" in i.lower()][0:8]
        d_h2o = [float(i.split("= ", 1)[-1].split()[0]) for i in lines if "d_h2o(" in i.lower()][0:8]
        e_h2o = [float(i.split("= ", 1)[-1].split()[0]) for i in lines if "e_h2o(" in i.lower()][0:8]
        f_h2o = [float(i.split("= ", 1)[-1].split()[0]) for i in lines if "f_h2o(" in i.lower()][0:8]
        g_h2o = [float(i.split("= ", 1)[-1].split()[0]) for i in lines if "g_h2o(" in i.lower()][0:8]
        awhite = [float(i.split("= ", 1)[-1].split()[0]) for i in lines if "awhite(" in i.lower()][0:8]
        aw = [float(i.split("= ", 1)[-1].split()[0]) for i in lines if "aw(" in i.lower()][0:8]
        bbw = [float(i.split("= ", 1)[-1].split()[0]) for i in lines if "bbw(" in i.lower()][0:8]
        oobwv = [float(i.split("= ", 1)[-1].split()[0]) for i in lines if "oobwv" in i.lower()][0:8]
        ooblw = [float(i.split("= ", 1)[-1].split()[0]) for i in lines if "ooblw" in i.lower()][0:8]
        wed = [float(i.split("= ", 1)[-1].split()[0]) for i in lines if "wed(" in i.lower()][0:8]
        waph = [float(i.split("= ", 1)[-1].split()[0]) for i in lines if "waph(" in i.lower()][0:8]
        Zia_tabel = [np.array(a_h2o), np.array(b_h2o), np.array(c_h2o), np.array(d_h2o), np.array(e_h2o),
                     np.array(f_h2o), np.array(g_h2o)]
        return (np.array(self.wavelength), np.array(F0), np.array(Tau_r), np.array(k_oz), np.array(t_co2),
                np.array(k_no2), Zia_tabel, np.array(awhite), np.array(aw), np.array(bbw), np.array(oobwv),
                np.array(ooblw),
                np.array(wed), np.array(waph))

    def get_image_info(self):
        bands = self.wavelength
        hysat = ["h1a", "h1b", "h1c"]
        if any(all(idxi in os.path.basename(self.infile).lower() for idxi in satsen) for satsen in list(itertools.product(hysat, ["oct"]))):
            date_str = os.path.basename(self.infile)[17:25]
            year, month, day = date_str[0:4], date_str[4:6], date_str[6:8]
            # 根据卫星传感器指定两个用于气溶胶估算的近红外波段，起始位0
            num_443, num_490, num_520, num_555, num_670, nirs_num, nirl_num = 1, 2, 3, 4, 5, 6, 7
            nwvis = 6
            red = num_670

            if 'L1B' in os.path.basename(self.infile):
                bands_str = ['L_' + str(i) for i in bands]
            elif 'L1A' in os.path.basename(self.infile):
                bands_str = ['DN_' + str(int(i)) for i in bands]
            else:
                bands_str = ['no sensor']
                print('no expected product level L1A or L1B, executing next file')
            h5f = h5py.File(self.infile, 'r')
            try:
                scales = h5f['/Calibration/Calibration Coefficients Scale factor'][()]
            except:
                scales = np.array([1., 1., 1., 1., 1., 1., 1., 1.])
            try:
                offsets = h5f['/Calibration/Calibration Coefficients Offsets factor'][()]
            except:
                offsets = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
            try:
                gains = h5f['/Calibration/Vicarious Calibration gain factor'][()]
                # 官方发布的H1C的替代定标系数是错的，零时全部改为1
                gains = np.array([1., 1., 1., 1., 1., 1., 1., 1.])
            except:
                gains = np.array([1., 1., 1., 1., 1., 1., 1., 1.])

            navi = h5f['/Navigation Data']
            lon_grid = preprodata(navi, 'Longitude')
            lat_grid = preprodata(navi, 'Latitude')

            vaa = h5f['/Navigation Data/Satellite Azimuth Angle'][()]
            vza = h5f['/Navigation Data/Satellite Zenith Angle'][()]
            try:
                sza = h5f['/Navigation Data/Solar Zenith Angle'][()]
                saa = h5f['/Navigation Data/Solar Azimuth Angle'][()]
            except:
                sza = h5f['/Navigation Data/Sun Zenith Angle'][()]
                saa = h5f['/Navigation Data/Sun Azimuth Angle'][()]

            data = np.empty(shape=(lon_grid.shape[0], lon_grid.shape[1], bands.__len__()))
            print("Calibration Coefficients:", "\n",
                  "scale:", scales, "\n",
                  "offset:", offsets, "\n",
                  "Vicarious Calibration:", gains)
            for i in range(bands.__len__()):
                data[:, :, i] = h5f['/Geophysical Data/' + bands_str[i]][()]
                data[:, :, i] = (data[:, :, i] * scales[i] + offsets[i]) * gains[i]  # 定标

        elif any(all(idxi in os.path.basename(self.infile).lower() for idxi in satsen) for satsen in list(itertools.product(hysat, ["czi"]))):
            pass
        elif all(idxi in os.path.basename(self.infile) for idxi in ['H1C', 'uvi']):
            pass
        elif all(idxi in os.path.basename(self.infile) for idxi in ['MOD']):
            pass
        elif all(idxi in os.path.basename(self.infile) for idxi in ['MYD']):
            pass
        elif all(idxi in os.path.basename(self.infile) for idxi in ['LC08_L1TP_']):
            # landsat 8 OLI输入的为一个文件夹
            from sensor.landsat8oli import read_landsat8oli
            [sza, vza, saa, vaa, lat_grid, lon_grid, data, year, month, day, num_443, num_490, num_520, num_555,
             num_670, nirs_num, nirl_num, nwvis, red] = read_landsat8oli.oli_info(self.infile)

        elif all(idxi in os.path.basename(self.infile) for idxi in ['LC08_oli_gee_phd']):
            # 自己从gee上下载的lc08 oli数据，不具备通用性，根据自己的情况进行设置
            from sensor.gee import read_custom_tif
            [sza, vza, saa, vaa, lat_grid, lon_grid, data, year, month, day, num_443, num_490, num_520, num_555,
             num_670, nirs_num, nirl_num, nwvis, red] = read_custom_tif.clipped_oli_info(self.infile)
        else:
            print('unidentified satellite sensor')

        # 范围裁剪
        if all([self.north, self.south, self.west, self.east]):
            loc = np.where(
                (self.south < lat_grid) & (lat_grid < self.north) & (self.west < lon_grid) & (lon_grid < self.east))
            # if loc[0].size < predefine.thresholds().pixels_num:  # 不到最小像元数量直接不计算了
            #     return ()
            up, low, left, right = np.min(loc[0]), np.max(loc[0]), np.min(loc[1]), np.max(loc[1])
            lat_grid = lat_grid[up:low, left:right]
            lon_grid = lon_grid[up:low, left:right]
            sza = sza[up:low, left:right]
            vza = vza[up:low, left:right]
            saa = saa[up:low, left:right]
            vaa = vaa[up:low, left:right]
            vza[vza > 88] = 88
            vza[vza < 0] = 0
            sza[sza > 88] = 88
            sza[sza < 0] = 0
            data = data[up:low, left:right, :]

        return [sza, vza, saa, vaa, lat_grid, lon_grid, data, year, month, day, num_443, num_490, num_520, num_555,
                num_670, nirs_num, nirl_num, nwvis, red]


def get_info(infile=None, lut_path=None, sensorID=None, mode="ac", north: float = None, south: float = None,
             west: float = None, east: float = None):
    info = base_info(infile=infile, sensorID=sensorID, lut_path=lut_path, mode=mode, north=north, south=south,
                     west=west, east=east)
    [lut_path, sensor_info, image_info] = info.run_main()
    print("look-up table: ", lut_path)
    return [lut_path, sensor_info, image_info]


if __name__ == '__main__':
    read_aerosol_info(infile=r'D:\test\ac2\manual_16_18_117_119\202101/T2021028021500.L2_LAC_OC')