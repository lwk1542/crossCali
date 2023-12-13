# -*- coding: utf-8 -*-
"""
@author: li wenkai
@contact: lwk1542@hotmail.com
@software: pycharm anaconda python3.7 
@file: vicarious_calibration.py
@time: 2021/10/31 17:38
@desc:
"""

import datetime

import numpy as np
import pandas as pd
try:
    import openpyxl
except:
    pass
from atmospheric_correction.oceancolor_acnirv2.sharepy import readfile
from ObservationGeometry import solar_zenith, solar_azimuth
from atmosphericCorrection.oceancolorACnirV2.share.seawifs.l2gen import aerosol_radV2, atmosphericParameter, gas_transmittance
from atmosphericCorrection.oceancolorACnirV2.share.seawifs.l2gen import whitecap_rad, rayleigh_rad, getglint

if __name__ == '__main__':

    # # 注意：与之前大气校正/交叉定标的程序不同，为了与seadas的格式一致，查找表路径从大气校正的LUT文件夹修改到share文件夹，传感器参数改为从msl12_sensor_info.dat中读取
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # 按照SVC_test.xlsx设置好实测数据的文件内容格式，指定传感器，

    insitu_file = "../data/SVC_test_2TRQH1B.xlsx"
    sensorID = "hy1bcocts"  # hy1bcocts modist

    # 两个近红外波段的位置
    nirs_num = 6
    nirl_num = 7
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # 1. 基础信息加载
    # # 查找表路径
    # rayleigh_lut_path, aerosol_lut_filepath, msl12_sensor_info_file = read_info.read_lut_path(instrument=sensorID)
    #
    # # 传感器参数
    # bands, Fo, Tau_r, k_oz, t_co2, k_no2, Zia_tabel, awhite, aw, bbw, oobwv, ooblw, wed, waph = \
    #     read_info.read_sensorinfo(sensorinfo_file=msl12_sensor_info_file)

    [lut_path, sensor_info, image_info] = readfile.get_info(sensorID=sensorID, mode="vc")
    (rayleigh_lut_path, aerosol_lut_path) = lut_path
    (bands, Fo, Tau_r, k_oz, t_co2, k_no2, Zia_table, awhite, aw, bbw, oobwv, ooblw, wed, waph) = sensor_info

    # 实测数据
    # bands_name, location_index, aot, rrs_w, latitude, longitude, year, month, day, hour, minute, second, \
    #     sensor_latitude, sensor_longitude, sensor_height = read_info.read_insitu(insitu_file=insitu_file)

    bands_name, location_index, aot, rrs_w, latitude, longitude, year, month, day, hour, minute, second, \
    vza, vaa = read_info.read_insitu2(insitu_file=insitu_file)

    # 观测几何: 计算观测几何(只支持矩阵运算)，如果是从影像中读取则使用另外的方法
    sza = solar_zenith.get_zenith(latitude, longitude, year, month, day, hour, minute, second)
    saa = solar_azimuth.get_azimuth(latitude, longitude, year, month, day, hour, minute, second)
    utctime = np.array([*map(
        lambda y, m, d, h, m_: datetime.datetime.strptime(str(y) + str(m) + str(d) + str(h) + str(m_), '%Y%m%d%H%M'),
        year, month, day, hour, minute)])
    # vza, vaa = geometry.get_observer_look(sensor_longitude, sensor_latitude, sensor_height, utctime, longitude,
    #                                       latitude, 0)

    # 2. 气象数据加载：根据海经纬度插值出相应的气象参数：风速、气压、 臭氧，其它如水汽柱也可输出
    # # 替代定标数据空间离散、日期离散，针对网络下载的气象数据，需逐点处理
    winds_peed = np.empty_like(latitude)
    winddirection = np.empty_like(winds_peed)
    pressure = np.empty_like(winds_peed)
    o3 = np.empty_like(winds_peed)
    rh = np.empty_like(winds_peed)
    water_vapor = np.empty_like(winds_peed)
    strat_no2 = np.empty_like(winds_peed)
    trop_no2 = np.empty_like(winds_peed)
    # 大气参数程序返回的这个taua是865nm的岂容胶光学厚度，本函数中并不使用这个变量
    taua = np.empty_like(winds_peed)
    doy = np.empty_like(winds_peed)
    for i in range(location_index.size):
        winds_peed[0, i], winddirection[0, i], pressure[0, i], o3[0, i], rh[0, i], water_vapor[0, i], strat_no2[0, i], \
        trop_no2[0, i], taua[0, i] = atmosphericParameter.get(Lon=longitude[:, i], Lat=latitude[:, i], year=year[i],
                                                              month=month[i], day=day[i],
                                                              time=str(hour[i]) + ":" + str(minute[i]))
        doy[0, i] = int(utctime[i].strftime('%j'))
    # # 日地距离调整因子，大气层顶辐照度修正
    A, B, C, D, E = 1.00014, 0.01671, 0.9856002831, 3.452868, 360.
    fsol = ((A - B * np.cos(2. * np.pi * (C * doy - D) / E) - 0.000014 * np.cos(4. * np.pi * (C * doy - D) / E)) ** 2)
    FoBAR = Fo.reshape(Fo.size, 1) * fsol

    # 3. 瑞利光学厚度的气压校正
    factor = pressure / 1013.25
    taur = factor.reshape(factor.shape[0], factor.shape[1], 1) * Tau_r.reshape(1, 1, -1)

    # 4. 大气透过率校正，臭氧从欧洲下载，臭氧消光截面不是从NASA下载，详情见函数内部。Mobely给的消光截面单位是错误的
    tg_sensor_o3, tg_solar_o3 = gas_transmittance.transmittance_o3(sza=sza, vza=vza, koz=k_oz, concentration=o3)
    tg_sensor_no2, tg_solar_no2 = gas_transmittance.transmittance_NO2(kno2=k_no2, sza=sza, vza=vza,
                                                                      strat_no2=strat_no2, trop_no2=trop_no2)
    tg_sensor_co2, tg_solar_co2 = gas_transmittance.transmittance_co2(t_co2=t_co2, sza=sza, vza=vza)
    tg_sensor_h2o, tg_solar_h2o = gas_transmittance.transmittance_h2o(water_vapor=water_vapor, sza=sza, vza=vza,
                                                                      zia_table=Zia_table)

    tg_sol = tg_solar_o3 * tg_solar_no2 * tg_sensor_co2 * tg_sensor_h2o  # 其它吸收暂时不考虑
    tg_sen = tg_sensor_o3 * tg_sensor_no2 * tg_solar_co2 * tg_solar_h2o

    # 5. 白帽反射
    rho_wc = whitecap_rad.calculate(U10=winds_peed, bands=bands)
    t_sen = np.empty(shape=(vza.shape[0], vza.shape[1], bands.size))
    t_sol = np.empty(shape=(vza.shape[0], vza.shape[1], bands.size))
    tLf = np.empty(shape=(vza.shape[0], vza.shape[1], bands.size))

    for i in range(bands.size):
        t_sen[:, :, i] = np.exp(-0.5 * (pressure / 1013.25) * taur[:, :, i] / np.cos(vza * np.pi / 180))
        t_sol[:, :, i] = np.exp(-0.5 * (pressure / 1013.25) * taur[:, :, i] / np.cos(sza * np.pi / 180))
        tLf[:, :, i] = rho_wc[:, :, i] * t_sol[:, :, i] * t_sen[:, :, i] * FoBAR[i] * np.cos(
            sza * np.pi / 180) / np.pi

    # 6. 瑞利贡献
    Lr = rayleigh_rad.rayleigh(rayleigh_lut_path=rayleigh_lut_path, sza=sza, vza=vza, saa=saa, vaa=vaa,
                               F0=FoBAR, windspeed=winds_peed, pressure=pressure, sensorID=sensorID)
    airmass1 = 1 / np.cos(sza * np.pi / 180) + 1 / np.cos(vza * np.pi / 180)

    # 7.氧气校正：只需要对750nm做，直接沿用了seawifs的校正系数，这一步很关键，影响很大，一般情况下，波段设置会避开氧气在这附近的吸收，故我没有乘。
    a_o2 = gas_transmittance.oxygen_ray(airmass1)
    t_o2 = 1.0 / gas_transmittance.oxygen_aer(airmass1)
    Lr[:, :, nirs_num] = Lr[:, :, nirs_num] * 1.  # * a_o2  #
    scaleRayleigh = 1.0 - np.exp(-798 / 10)  # 传感器高度校正，这个影响很小
    # Lr = Lr * scaleRayleigh.reshape(scaleRayleigh.shape[0], scaleRayleigh.shape[1], 1)

    # 7.气溶胶反射: 根据测量的各波段气溶胶光学厚度确定气溶胶模型并计算气溶胶反射率
    # aerosol = aerosol_radV2.fixdaot(aot=aot, relative_humidity=rh, sza=sza, vza=vza, saa=saa, vaa=vaa,
    #                                 wave=bands, iwnir_s=nirs_num, iwnir_l=nirl_num, taur=Tau_r, pressure=pressure,
    #                                 aerosol_lut_filepath=aerosol_lut_path)
    aerosol = aerosol_radV2.fixed_2bands_aot(aot=aot, relative_humidity=rh, sza=sza, vza=vza, saa=saa, vaa=vaa,
                                            wave=bands, iwnir_s=nirs_num, iwnir_l=nirl_num, taur=Tau_r, pressure=pressure,
                                            aerosol_lut_filepath=aerosol_lut_path)

    taua, rhoax, t_sensorx, t_solarx = aerosol
    #
    rhoa = np.empty_like(Lr)
    t_sensor = np.empty_like(Lr)
    t_solar = np.empty_like(Lr)
    F0_t = np.empty_like(Lr)
    rhoa[0, :, :] = rhoax.T
    t_sensor[0, :, :] = t_sensorx.T
    t_solar[0, :, :] = t_solarx.T
    F0_t[0, :, :] = FoBAR.T
    # # 反射率转为福亮度
    La = rhoa / (np.pi / F0_t / np.cos(sza.reshape(sza.shape[0], sza.shape[1], 1) * np.pi / 180))

    # 8. 耀斑反射
    wd_rad = winddirection * np.pi / 180
    # 由于涉及到F0参数的矩阵形状问题，无法直接使用原来的耀斑校正函数；如果修改耀斑计算函数，又涉及到大气校正和交叉定标中相关函数的修改
    # 故这里使用一个循环来避免这些问题。暂时留在这里，以后再修改
    TLg = np.empty_like(La)
    for i in range(location_index.size):
        TLg[0, i, :] = getglint.main_exec(sza=sza[:, i].reshape(1, 1), vza=vza[:, i].reshape(1, 1),
                                          vaa=vaa[:, i].reshape(1, 1), saa=saa[:, i].reshape(1, 1),
                                          taur=taur[:, i, :].reshape(1, 1, bands.size),
                                          La=La[:, i, :].reshape(1, 1, bands.size), F0=F0_t[0, i, :],
                                          windspeed=winds_peed[:, i].reshape(1, 1),
                                          winddirection=wd_rad[:, i].reshape(1, 1),
                                          taua=aot[:, i, :].reshape(1, 1, bands.size), iter_num=1, mode=2)

    # 9. 水体贡献
    nLw = rrs_w * F0_t
    csza = np.zeros(shape=(sza.shape[0], sza.shape[1], 1))
    csza[:, :, 0] = np.cos(sza * np.pi / 180)
    Lw = nLw * t_solar * tg_sol * csza / fsol.reshape(1, fsol.size, 1)
    tLw = Lw / tg_sol * t_sensor

    # 10. Ltoa
    Lt_simu = tLf + Lr + TLg + tLw+La

    # 11. 导出计算结果
    df = pd.DataFrame(Lt_simu[0, :, :], columns=bands_name, index=location_index)
    try:
        # 在不同版本的pandas下，写出到excel命令可能会覆盖掉本来的数据表，因此，请备份原始数据
        with pd.ExcelWriter(insitu_file, engine='openpyxl', mode='a') as writer:
            df.to_excel(writer, sheet_name="Ltoa_simulation_" + sensorID,  index=True)
    except:
        # 无法导出到excel，则导出到csv，通常可能是缺少openxyl库
        df.to_csv(insitu_file+".csv", index=True)