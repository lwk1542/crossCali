# -*- coding: utf-8 -*-
"""
@author: li wenkai
@contact: lwk1542@hotmail.com
@software: pycharm anaconda python3.7 
@file: atmosphericParameter.py
@time: 2021/1/22 14:41
@desc: 将获取的风速、气压和臭氧、NO2数据、rh插值为匹配影像大小的矩阵.还包括平均的气溶胶光学厚度
    默认从ERA下载，备选多年统计值，可以从NASA ocean color爬取下载（有程序，但是没有列入下载选项）

    把所有的经度坐标调整到0-360度范围
"""
import datetime
import os
import re

import cdsapi
import h5py
import numpy as np
import urllib3
from netCDF4 import Dataset
from scipy import interpolate


def alternative(year: int = 2010, month: int = 6, day: int = 15):
    """
    使用预定义的气象参数
    Parameters
    ----------
    year :
    month :
    day :

    Returns
    -------

    """
    # dirname, filename /= os.path.split(os.path.abspath(sys.argv[0]))
    year = int(year)
    month = int(month)
    day = int(day)
    metfile = r"../share/common/met_climatology_v2014.h5"
    month_dict = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'August',
                  9: 'September', 10: 'October', 11: 'November', 12: 'December'}
    f = h5py.File(metfile, 'r')  # 读NECP文件
    group = f['/' + month_dict.get(month)]
    pr = group['press_mean'][()]  # 单位 pa
    rh = group['rel_hum_mean'][()]  # 单位 pa
    wsv = group['z_wind_mean'][()]  # 单位 m/s
    wsu = group['m_wind_mean'][()]  # 单位 m/s
    pw = group['p_water_mean'][()] / 10  # 单位 /* to make the kg m^-2 to g cm^-2 */

    latmet = np.array(range(90, -91, -1))
    lonmet = np.array(range(-180, 180, 1))
    f.close()
    date = datetime.datetime.strptime(str(year) + str(month) + str(day), '%Y%m%d')
    doy = date.strftime('%j')
    o3file = r"../share/common/ozone_climatology_v2014.h5"
    f = h5py.File(o3file, 'r')  # 读NECP文件
    try:
        o3 = f['/Geophysical Data/ozone_mean_' + doy][()]
    except IOError:
        o3 = np.array([0])
        print(
            'can not read ozone file variable of ozone_climatology_v2014.h5 : ' + '/Geophysical Data/ozone_mean_' + doy)
    # o3_2 = o3 * 2.6867e16  # molecules per centimeter square
    # /* convert from Dobson units to atm-cm */
    o3_2 = o3 / 1000

    lato3 = np.array(range(90, -90, -1)) - 0.5
    lono3 = np.array(range(-180, 180, 1)) + 0.5

    return latmet, lonmet, pr, wsv, wsu, rh, pw, lato3, lono3, o3_2


def read_no2(month: int = 6):
    """
    no2是静态数据，采用多年均值
    Parameters
    ----------
    month :

    Returns
    -------

    """
    month = int(month)
    metfile = r"../share/common/no2_climatology_v2013.h5"
    month_dict = {1: '01', 2: '02', 3: '03', 4: '04', 5: '05', 6: '06', 7: '07', 8: '08',
                  9: '09', 10: '10', 11: '11', 12: '12'}
    f = h5py.File(metfile, 'r')  # 读NECP文件
    strat_no2 = f['/Geophysical Data/strat_no2_' + month_dict.get(month)][()] * 1e15
    tot_no2 = f['/Geophysical Data/tot_no2_' + month_dict.get(month)][()] * 1e15
    trop_no2 = f['/Geophysical Data/trop_no2_' + month_dict.get(month)][()] * 1e15

    lat = np.array(np.arange(-89.875, 90, 0.25))
    lon = np.array(np.arange(179.875, -180, -0.25))
    f.close()

    return lat, lon, strat_no2, tot_no2, trop_no2


def ecmwf_download(year=None, month=None, day=None, time=None, path='common'):
    """
    下载气象数据：https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form
    Returns:
    """

    ofile = path + os.sep + 'ECMWF' + str(year) + str(month) + str(day) + str(time[0:2]) + '.nc'
    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                '10m_u_component_of_wind', '10m_v_component_of_wind', 'mean_sea_level_pressure',
                'total_column_ozone', 'total_column_water_vapour', 'total_precipitation',
            ],
            'year': str(year),
            'month': str(month),
            'day': str(day),
            'time': str(time),
        },
        ofile)
    print('Atmospheric environment anxiliary data is:' + ofile)

    # """
    #     下载相对湿度：https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form
    # """
    ofile1 = path + os.sep + 'ECMWF' + str(year) + str(month) + str(day) + str(time[0:2]) + '_relativeHumidity.nc'
    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'pressure_level': '1000',
            'year': str(year),
            'month': str(month),
            'day': str(day),
            'time': str(time),
            'variable': 'relative_humidity',
        },
        ofile1)
    print('Atmospheric environment relative humidity data is:' + ofile1)

    return ofile, ofile1


def read_ecmwf(ofile=None, ofile1=None):
    f = Dataset(ofile)  # 读NECP文件
    pr = f.variables['msl'][()][0]  # 单位 pa
    pr = pr / 100  # 将单位转化为hpa
    wv = f.variables['tcwv'][()][0] / 10  # 单位 /* to make the kg m^-2 to g cm^-2 */
    wsv = f.variables['v10'][()][0]  # 单位 m/s
    wsu = f.variables['u10'][()][0]  # 单位 m/s
    o3 = f.variables['tco3'][()][0]  # 单位 kg/m2
    latmet = f.variables['latitude'][()]
    lonmet = f.variables['longitude'][()]
    o3_1 = o3 / 2.1415E-5  # Dobson Unit   https://sacs.aeronomie.be/info/dobson.php
    # o3_2 = o3_1 * 2.6867e16  # molecules per centimeter square
    o3_2 = o3_1 / 1000  # /* convert from Dobson units to atm-cm */
    f.close()
    f1 = Dataset(ofile1)  # 读NECP文件
    rh = f1.variables['r'][()][0]
    latrh = f1.variables['latitude'][()]
    lonrh = f1.variables['longitude'][()]
    f1.close()
    return latmet, lonmet, pr, wv, wsv, wsu, latmet, lonmet, o3_2, latrh, lonrh, rh


def download_from_nasa(year='2002', month='06', day='15', path=''):
    """
    https://oceancolor.gsfc.nasa.gov/docs/ancillary/
    https://oceandata.sci.gsfc.nasa.gov/Ancillary/Meteorological/
    """

    # 这个没有完成

    # o3
    http = urllib3.PoolManager()  # 创建PoolManager对象生成请求, 由该实例对象处理与线程池的连接以及线程安全的所有细节
    # get方式请求
    response = http.request('GET',
                            'https://acd-ext.gsfc.nasa.gov/anonftp/toms/omi/data/ozone/Y2020/L3_ozone_omi_' +
                            year + month + day + '.txt')
    # print(response.status,response.data.decode('utf-8'))  # 获得状态码, html源码(utf-8解码)
    # with open("D:\Data"+os.sep+"test.txt", "w") as f:
    #     f.write(response.data.decode('utf-8'))            # 这句话自带文件关闭功能，不需要再写f.close()
    # print()
    o3 = np.zeros(shape=(180, 360))
    if response.status == 200:
        x = re.split('\n', response.data.decode('utf-8'))
        for i in range(180):
            print(i)
            # 取出每一个纬度的数据，
            xx = [k[1:] for k in x[3 + i * 15:3 + (i + 1) * 15]]
            str1 = ''
            data_row = str1.join(xx)
            st2 = re.findall(r'.{3}', data_row)[0:360]  # 360后面的是注释
            row = np.array([int(c) for c in st2])
            o3[179 - i, :] = row
        lon = np.linspace(-179.5, 179.5, num=360, endpoint=True, retstep=False, dtype=None).reshape(1, -1)
        lat = np.linspace(89.5, -89.5, num=180, endpoint=True, retstep=False, dtype=None).reshape(-1, 1)
        h5 = h5py.File(path + os.sep + 'L3_ozone_omi_' + year + month + day + '.h5', 'w')
        o3_ = h5.create_dataset('ozone', (o3.shape[0], o3.shape[1]), dtype='f', data=o3)
        h5.create_dataset('latitude', (lat.shape[0], lat.shape[1]), dtype='f', data=lat)
        h5.create_dataset('longitude', (lat.shape[0], lat.shape[1]), dtype='f', data=lat)
        o3.attres.creat('unit', 'Dobson units')
        h5.close()
    # met
    response = http.request('GET',
                            'https://acd-ext.gsfc.nasa.gov/anonftp/toms/omi/data/ozone/Y2020/L3_ozone_omi_' + year + month + day + '.txt')
    doy = datetime.datetime.strptime(year + month + day, '%Y%m%d').strftime('%j')
    http = 'https://oceandata.sci.gsfc.nasa.gov/Ancillary/Meteorological/' + year + '/' + doy + '/' + "N" + year + doy + '02_MET_NCEPR2_6h.hdf'


def interp(lat=None, lon=None, value=None, Lon=None, Lat=None):
    # 比输入数据的范围稍大
    Lat_min = np.floor(np.min(Lat))-1
    Lat_max = np.ceil(np.max(Lat))+1
    Lon_min = np.floor(np.min(Lon))-1
    Lon_max = np.ceil(np.max(Lon))+1
    # print(Lat_min,Lat_max,Lon_min,Lon_max)

    # 网格化处理
    # llat = np.tile(llat.reshape(-1, 1), (1, llon.size))
    # llon = np.tile(llon.reshape(1, -1), (llat.size, 1))
    llon, llat = np.meshgrid(lon, lat)

    loc = np.where((Lat_min <= llat) & (llat <= Lat_max) & (Lon_min <= llon) & (llon <= Lon_max))
    up, low, left, right = np.min(loc[0])-1, np.max(loc[0])+1, np.min(loc[1])-1, np.max(loc[1])+1
    llat = llat[up:low, left:right]
    llon = llon[up:low, left:right]
    value = value[up:low, left:right]

    # 经度从-180----180的坐标转换为 0-360度
    # Lon[Lon < 0] = Lon[Lon < 0] + 360
    # llon[llon< 0] = llon[llon < 0] + 360

    return interpolate.griddata((llat.flatten(), llon.flatten()), value.flatten(), (Lat, Lon), method='nearest')


def initial_taua(month: str = '06', Lat=None, Lon=None):
    """
    各地月均气溶胶光学厚度，使用这个气溶胶光学厚度去估算耀斑
    Parameters
    ----------
    month :
    Lat :
    Lon :
    Returns
    -------
    """

    f = h5py.File(r"../share/common/taua865_climatology.h5", mode='r')
    taua_865 = f['data' + month][()] * 0.005
    lon = f['Longitude'][()]
    lat = f['Latitude'][()]
    # 网格化处理
    llat = np.tile(lat.reshape(-1, 1), (1, lon.size))
    llon = np.tile(lon.reshape(1, -1), (lat.size, 1))

    # 比输入数据的范围稍大
    Lat_min = np.floor(np.min(Lat))
    Lat_max = np.ceil(np.max(Lat))
    Lon_min = np.floor(np.min(Lon))
    Lon_max = np.ceil(np.max(Lon))
    # 都是
    if Lon_min > 180:
        Lon_min, Lon_max = Lon_min-360, Lon_max-360
    if Lat.size == 1:
        # 单点数据，直接插值，平面数据也可以直接插值，而不用裁剪共同区域。但是这里暂且没执行这样的操作。需要评估griddata的插值效率
        taua_865 = interpolate.griddata((llat.flatten(), llon.flatten()), taua_865.flatten(), (Lat, Lon),
                                        method='nearest')
    elif (Lon_max == Lon_min) and (Lat_max == Lat_min):
        taua_865 = interpolate.griddata((llat.flatten(), llon.flatten()), taua_865.flatten(), (Lat, Lon),
                                        method='nearest')
    else:
        # 在一个有限的范围内进行插值
        loc = np.where((Lat_min <= llat) & (llat <= Lat_max) & (Lon_min <= llon) & (llon <= Lon_max))
        up, low, left, right = np.min(loc[0]), np.max(loc[0]), np.min(loc[1]), np.max(loc[1])
        llat = llat[up:low, left:right]
        llon = llon[up:low, left:right]
        taua_865 = taua_865[up:low, left:right]
        taua_865 = interpolate.griddata((llat.flatten(), llon.flatten()), taua_865.flatten(), (Lat, Lon),
                                        method='nearest')
        taua_865[np.isnan(taua_865)] = -10
        taua_865[taua_865 < 1e-6] = np.nan
        taua_865[np.isnan(taua_865)] = np.nanmean(taua_865)
    return taua_865


def get(Lon=None, Lat=None, year='2003', month='06', day='15', time='03:00'):
    """
    获取大气参数
    """
    year = str(year)
    month = str(month)
    if len(month) == 1:
        month = "0" + month
    day = str(day)
    if len(day) == 1:
        day = "0" + day
    if len(time) == 4:
        # 分钟只有一位显示
        time = "0" + time
    ofile = 'common' + os.sep + 'ECMWF' + str(year) + str(month) + str(day) + str(time[0:2]) + '.nc'
    ofile1 = 'common' + os.sep + 'ECMWF' + str(year) + str(month) + str(day) + str(time[0:2]) + '_relativeHumidity.nc'
    # 检查预定义的气象参数
    latmet, lonmet, pr, wsv, wsu, rh, pw, lato3, lono3, o3 = alternative(year=year, month=month, day=day)
    # Lon[Lon > 180] = Lon[Lon > 180] - 360
    # if all([os.path.exists(ofile), os.path.exists(ofile1)]):
    #     print('Meteorological data exist...')
    #     latmet, lonmet, pr, pw, wsv, wsu, lato3, lono3, o3, latrh, lonrh, rh = read_ecmwf(ofile=ofile, ofile1=ofile1)
    # else:
    #     try:
    #         # urllib3.request.urlopen('https://baidu.com', timeout=10)           # 如果网络连接通畅
    #         print('download Meteorological auxiliary data from ECMWF...')
    #         ofile, ofile1 = ecmwf_download(year=year, month=month, day=day, time=time,
    #                                        path=os.getcwd() + os.sep + 'common')
    #         # 这里需要修改
    #         latmet, lonmet, pr, pw, wsv, wsu, lato3, lono3, o3, latrh, lonrh, rh = read_ecmwf(ofile=ofile,
    #                                                                                           ofile1=ofile1)
    #     except:
    #         print('can not download Meteorological auxiliary data from ECMWF...')
    #         print('using predefinded Meteorological auxiliary data...')
    #         latmet, lonmet, pr, wsv, wsu, rh, pw, lato3, lono3, o3 = alternative(year=year, month=month, day=day)

    windspeed = np.sqrt(wsv ** 2 + wsu ** 2)  # 单位 m/s
    windspeed = interp(lat=latmet, lon=lonmet, value=windspeed, Lon=Lon, Lat=Lat)
    windspeed[windspeed < 0] = 0
    # 风向
    # winddirection = (90 - (np.arctan2(wsu, wsv) * 180 / np.pi + 180)) * np.pi / 180
    winddirection = np.arctan2(wsu, wsv) * 180 / np.pi
    winddirection = interp(lat=latmet, lon=lonmet, value=winddirection, Lon=Lon, Lat=Lat)
    pressure = interp(lat=latmet, lon=lonmet, value=pr, Lon=Lon, Lat=Lat)
    pressure[pressure < 900] = 900
    pressure[pressure > 1100] = 1100
    reality_humidity = interp(lat=latmet, lon=lonmet, value=rh, Lon=Lon, Lat=Lat)
    water_vapor = interp(lat=latmet, lon=lonmet, value=pw, Lon=Lon, Lat=Lat)
    o3 = interp(lat=lato3, lon=lono3, value=o3, Lon=Lon, Lat=Lat)
    o3[o3 < 0] = 0
    # NO2
    lat, lon, strat_no2, tot_no2, trop_no2 = read_no2(month=month)  # NO2使用静态数据
    strat_no2 = interp(lat=lat, lon=lon, value=strat_no2, Lon=Lon, Lat=Lat)
    # tot_no2 = intper2(lat_=lat, lon_=lon, value=tot_no2, Lon=Lon, Lat=Lat)
    trop_no2 = interp(lat=lat, lon=lon, value=trop_no2, Lon=Lon, Lat=Lat)
    strat_no2[strat_no2 < 0] = 0
    strat_no2[strat_no2 < 0] = 0

    # 气溶胶光学厚度单独处理
    taua_865_init = initial_taua(month=month, Lat=Lat, Lon=Lon)

    return windspeed, winddirection, pressure, o3, reality_humidity, water_vapor, strat_no2, trop_no2, taua_865_init

# if __name__ == "__main__":
#     download_o3_from_nasa(date='20200601', path=r'D:\Data')