# -*- coding: utf-8 -*-
"""
@author: li wenkai
@contact: lwk1542@hotmail.com
@software: pycharm anaconda python3.7 
@file: read_img_info.py
@time: 2021/10/31 20:56
@desc:
"""
import os


def read_insitu(insitu_file="SVC_test.xlsx"):
    import numpy as np
    import pandas as pd
    """
    Args:
        insitu_file ():

    Returns:
        各波段气溶胶光学厚度，遥感反射率，纬度，经度，日期，时间
    """
    aot_df = pd.read_excel(insitu_file, sheet_name="aot", index_col=0, header=0)
    rrs_df = pd.read_excel(insitu_file, sheet_name="rrs", index_col=0, header=0)
    auxiliary_df = pd.read_excel(insitu_file, sheet_name="auxiliary", index_col=0, header=0)
    satellite_position_df = pd.read_excel(insitu_file, sheet_name="satellite_parameters", index_col=0, header=0)

    bands_name = rrs_df.columns
    location_index = rrs_df.index

    rows, columns = aot_df.shape[0], aot_df.shape[1]
    aot = np.empty(shape=(1, rows, columns))
    rrs = np.empty_like(aot)
    latitude = np.empty(shape=(1, rows))
    longitude = np.empty(shape=(1, rows))
    # time = np.empty(shape=(rows),dtype="S18")
    year = np.empty(shape=(rows), dtype=int)
    month = np.empty(shape=(rows), dtype=int)
    day = np.empty(shape=(rows), dtype=int)
    hour = np.empty(shape=(rows), dtype=int)
    minute = np.empty(shape=(rows), dtype=int)
    second = np.empty(shape=(rows), dtype=int)
    sensor_latitude = np.empty(shape=(1, rows))
    sensor_longitude = np.empty(shape=(1, rows))
    sensor_height = np.empty(shape=(1, rows))

    for i, ind in enumerate(auxiliary_df.index):
        # 逐点读取
        aot_temp = np.array(aot_df.loc[ind, :]).reshape(1, 1, -1)
        rrs_temp = np.array(rrs_df.loc[ind, :]).reshape(1, 1, -1)
        auxiliary_temp = auxiliary_df.loc[ind, :]
        satellite_position_temp = satellite_position_df.loc[ind, :]

        aot[0, i, :] = aot_temp
        rrs[0, i, :] = rrs_temp
        latitude[0, i] = auxiliary_temp.loc["latitude"]
        longitude[0, i] = auxiliary_temp.loc["longitude"]
        time_temp = auxiliary_temp.loc["utc-time"]
        year[i] = int(time_temp[0:4])
        month[i] = int(time_temp[5:7])
        day[i] = int(time_temp[8:10])
        hour[i] = int(time_temp[11:13])
        minute[i] = int(time_temp[14:])
        second[i] = 0

        sensor_latitude[0, i] = satellite_position_temp.loc["latitude"]
        sensor_longitude[0, i] = satellite_position_temp.loc["longitude"]
        sensor_height[0, i] = satellite_position_temp.loc["height(km)"]

    return bands_name, location_index, aot, rrs, latitude, longitude, year, month, day, hour, minute, second, \
           sensor_latitude, sensor_longitude, sensor_height


def read_insitu2(insitu_file="SVC_test.xlsx"):
    import numpy as np
    import pandas as pd
    """
    Args:
        insitu_file ():

    Returns:
        各波段气溶胶光学厚度，遥感反射率，纬度，经度，日期，时间
    """
    aot_df = pd.read_excel(insitu_file, sheet_name="aot", index_col=0, header=0)
    rrs_df = pd.read_excel(insitu_file, sheet_name="rrs", index_col=0, header=0)
    auxiliary_df = pd.read_excel(insitu_file, sheet_name="auxiliary", index_col=0, header=0)
    satellite_position_df = pd.read_excel(insitu_file, sheet_name="satellite_parameters", index_col=0, header=0)

    bands_name = rrs_df.columns
    location_index = rrs_df.index

    rows, columns = aot_df.shape[0], aot_df.shape[1]
    aot = np.empty(shape=(1, rows, columns))
    rrs = np.empty_like(aot)
    latitude = np.empty(shape=(1, rows))
    longitude = np.empty(shape=(1, rows))
    # time = np.empty(shape=(rows),dtype="S18")
    year = np.empty(shape=(rows), dtype=int)
    month = np.empty(shape=(rows), dtype=int)
    day = np.empty(shape=(rows), dtype=int)
    hour = np.empty(shape=(rows), dtype=int)
    minute = np.empty(shape=(rows), dtype=int)
    second = np.empty(shape=(rows), dtype=int)
    vza = np.empty(shape=(1, rows))
    vaa = np.empty(shape=(1, rows))
    # sensor_height = np.empty(shape=(1, rows))

    for i, ind in enumerate(auxiliary_df.index):
        # 逐点读取
        aot_temp = np.array(aot_df.loc[ind, :]).reshape(1, 1, -1)
        rrs_temp = np.array(rrs_df.loc[ind, :]).reshape(1, 1, -1)
        auxiliary_temp = auxiliary_df.loc[ind, :]
        satellite_position_temp = satellite_position_df.loc[ind, :]

        aot[0, i, :] = aot_temp
        rrs[0, i, :] = rrs_temp
        latitude[0, i] = auxiliary_temp.loc["latitude"]
        longitude[0, i] = auxiliary_temp.loc["longitude"]
        time_temp = auxiliary_temp.loc["utc-time"]
        year[i] = int(time_temp[0:4])
        month[i] = int(time_temp[5:7])
        day[i] = int(time_temp[8:10])
        hour[i] = int(time_temp[11:13])
        minute[i] = int(time_temp[14:])
        second[i] = 0

        vza[0, i] = satellite_position_temp.loc["vza"]
        vaa[0, i] = satellite_position_temp.loc["vaa"]
        # sensor_height[0, i] = satellite_position_temp.loc["height(km)"]

    return bands_name, location_index, aot, rrs, latitude, longitude, year, month, day, hour, minute, second, \
           vza, vaa


def get(infile=None, sensor_id=None, block_size: int = None):
    if sensor_id == "sdgsat1mii":  # 传感器
        if all(idx in os.path.basename(infile) for idx in ['KX10_MII_', "L4A"]):  # 数据级别
            from sdgsat1mii import read_sdgsat1mii
            image_info = read_sdgsat1mii.get(infile, block_size)
            [sza, vza, saa, vaa, gains, bias, data_Iterator, year, month, day, num_443, num_490, num_520, num_555,
             num_670, nirs_num, nirl_num, nwvis, red] = image_info   # data_Iterator是一个迭代器，包含有多个参数
            return [sza, vza, saa, vaa, gains, bias, data_Iterator, year, month, day, num_443, num_490, num_520,
                    num_555, num_670, nirs_num, nirl_num, nwvis, red]

    elif sensor_id == 'hy1acocts':
        pass
    elif sensor_id == 'hy1bcocts':
        pass
    elif sensor_id == 'hy1ccocts':
        pass
    elif sensor_id == 'lc08oli':
        pass
    elif sensor_id == 'modist':
        pass
    elif sensor_id == 'modisa':
        pass
    elif sensor_id == 'seawifsphd':
        pass
    else:
        pass


if __name__ == '__main__':
    read_insitu()
