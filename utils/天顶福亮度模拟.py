# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: li wenkai
@contact: lwk1542@hotmail.com
@software: pycharm anaconda python3.7
@file: 地面差异引起的天顶差异.py
@time: 2021/12/21 16:54
@desc:
"""

import itertools
import numpy as np
import pandas as pd

try:
    import openpyxl
except:
    pass
from sensor import read_img_info
from atmosphericCorrection.oceancolorACnirV2.share.seawifs.l2gen import aerosol_radV2, gas_transmittance
from atmosphericCorrection.oceancolorACnirV2.share.seawifs.l2gen import whitecap_rad, rayleigh_rad


class simula(object):

    def __init__(self):
        return

    def run_main(self):
        aot, rrs, sza, vza, reaa = self.generate_data()
        self.rt(aot, rrs, sza, vza, reaa)
        return

    def generate_data(self):
        # 不同波段反射率的范围
        bands_name = ["412nm", "443nm", "490nm", "520nm", "565nm", "670nm", "750nm", "865nm"]
        Rrs = np.arange(0.0, 0.051, 0.0005)
        aot = np.arange(0.1 - 0.004, 0.1 + 0.004, 0.001)
        # vza=np.arange(0., 80, 10)
        # sza = np.arange(0., 80, 10)
        # reaa=np.arange(0., 180, 20)
        vza = np.array([50])
        sza = np.array([50])
        reaa = np.array([45])

        variableNum = list(itertools.product(Rrs, aot, vza, sza, reaa))
        rrs_temp = [i[0] for i in variableNum]
        aot_temp = [i[1] for i in variableNum]
        rrs_temp = np.array(rrs_temp).reshape(1, -1, 1)
        rrs_input = np.tile(rrs_temp, (1, 1, 8))
        aot_temp = np.array(aot_temp).reshape(1, -1, 1)
        aot_input = np.tile(aot_temp, (1, 1, 8))

        vza_input = [i[2] for i in variableNum]
        sza_input = [i[3] for i in variableNum]
        reaa_input = [i[4] for i in variableNum]
        # dic = {"aot": aot_input,
        #        "rrs": rrs_input,
        #        "vza": vza_input,
        #        "sza": sza_input,
        #        "reaa": reaa_input}
        # df = pd.DataFrame(dic)
        # simupara = r"D:\researchProject\论文-毕业论文/calibration_site_svc.xlsx"
        # with pd.ExcelWriter(simupara, engine='openpyxl', mode='a') as writer:
        #     df.to_excel(writer, sheet_name="para", index=True)

        return aot_input, rrs_input, np.array(sza_input).reshape(1, -1), \
               np.array(vza_input).reshape(1, -1), np.array(reaa_input).reshape(1, -1)

    def rt(self, aot, rrs_w, sza, vza, reaa):
        # # 注意：与之前大气校正/交叉定标的程序不同，为了与seadas的格式一致，查找表路径从大气校正的LUT文件夹修改到share文件夹，传感器参数改为从msl12_sensor_info.dat中读取
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 按照SVC_test.xlsx设置好实测数据的文件内容格式，指定传感器，

        insitu_file = r"D:\researchProject\论文-毕业论文/simulatedLtoa.xlsx"
        sensorID = "hy1ccocts"
        sensor_height = 800

        # 两个近红外波段的位置
        nirs_num = 6
        nirl_num = 7
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 1. 基础信息加载
        # 查找表路径
        rayleigh_lut_path, aerosol_lut_filepath, msl12_sensor_info_file = read_img_info.read_lut_path(instrument=sensorID)

        # 传感器参数
        bands, Fo, Tau_r, k_oz, t_co2, k_no2, Zia_tabel, awhite, aw, bbw, oobwv, ooblw, wed, waph = \
            read_img_info.read_sensorinfo(sensorinfo_file=msl12_sensor_info_file)

        # 实测数据
        # aot, rrs, sza, vza, reaa = read_info.read_insitu2(insitu_file=insitu_file)

        # 2. 气象数据加载：根据海经纬度插值出相应的气象参数：风速、气压、 臭氧，其它如水汽柱也可输出
        # # 替代定标数据空间离散、日期离散，针对网络下载的气象数据，需逐点处理

        (rows, cols, deeps) = aot.shape
        winds_peed = np.zeros(shape=(rows, cols))
        winddirection = np.zeros_like(winds_peed)
        pressure = np.ones_like(winds_peed) * 1013.15
        o3 = np.ones_like(winds_peed) * 0.25
        modnum = np.ones_like(winds_peed, dtype=int) * 10
        rh = np.empty_like(winds_peed) * 80
        water_vapor = np.ones_like(winds_peed) * 33
        strat_no2 = np.ones_like(winds_peed) * 3.11e+15
        trop_no2 = np.ones_like(winds_peed) * 0.19e+15
        # 大气参数程序返回的这个taua是865nm的岂容胶光学厚度，本函数中并不使用这个变量
        taua = np.ones_like(winds_peed) * 0.1
        doy = np.ones_like(winds_peed) * 180
        longitude = np.ones_like(winds_peed) * -158
        latitude = np.ones_like(winds_peed) * 19.4
        year = 2003
        month = 6
        day = 15
        hour = 2
        minute = 30

        # # 日地距离调整因子，大气层顶辐照度修正
        # doy = 180
        A, B, C, D, E = 1.00014, 0.01671, 0.9856002831, 3.452868, 360.
        fsol = ((A - B * np.cos(2. * np.pi * (C * doy - D) / E) - 0.000014 * np.cos(
            4. * np.pi * (C * doy - D) / E)) ** 2)
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
                                                                          zia_table=Zia_tabel)

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
        print("a")

        # 6. 瑞利贡献
        Lr = rayleigh_rad.rayleigh(rayleigh_lut_path=rayleigh_lut_path, sza=sza, vza=vza, reaa=reaa,
                                   F0=FoBAR, windspeed=winds_peed, pressure=pressure, sensorID=sensorID)
        airmass1 = 1 / np.cos(sza * np.pi / 180) + 1 / np.cos(vza * np.pi / 180)

        # 7.氧气校正：只需要对750nm做，直接沿用了seawifs的校正系数，这一步很关键，影响很大，一般情况下，波段设置会避开氧气在这附近的吸收，故我没有乘。
        a_o2 = gas_transmittance.oxygen_ray(airmass1)
        t_o2 = 1.0 / gas_transmittance.oxygen_aer(airmass1)
        Lr[:, :, nirs_num] = Lr[:, :, nirs_num] * 1.  # * a_o2  #
        scaleRayleigh = 1.0 - np.exp(-sensor_height / 10)  # 传感器高度校正，这个影响很小
        Lr = Lr * scaleRayleigh  # .reshape(scaleRayleigh.shape[0], scaleRayleigh.shape[1], 1)
        aerosol = aerosol_radV2.fixd_aer_aot(aot=aot, modnum=modnum, sza=sza, vza=vza, reaa=reaa,
                                             aerosol_lut_filepath=aerosol_lut_filepath, taur=Tau_r, pressure=pressure)
        rhoax, t_sensorx, t_solarx = aerosol

        rhoa = np.empty_like(Lr)
        t_sensor = np.empty_like(Lr)
        t_solar = np.empty_like(Lr)
        F0_t = np.empty_like(Lr)
        rhoa[0, :, :] = rhoax[0, :, :]
        t_sensor[0, :, :] = t_sensorx.T
        t_solar[0, :, :] = t_solarx.T
        F0_t[0, :, :] = FoBAR.T
        # # 反射率转为福亮度
        La = rhoa / (np.pi / F0_t / np.cos(sza.reshape(sza.shape[0], sza.shape[1], 1) * np.pi / 180))

        # 8. 耀斑反射
        # 定标假设没有耀斑
        # wd_rad = winddirection * np.pi / 180
        # # 由于涉及到F0参数的矩阵形状问题，无法直接使用原来的耀斑校正函数；如果修改耀斑计算函数，又涉及到大气校正和交叉定标中相关函数的修改
        # # 故这里使用一个循环来避免这些问题。暂时留在这里，以后再修改
        # TLg = np.empty_like(La)
        #
        # TLg[0, i, :] = getglint.main_exec(sza=sza[:, i].reshape(1, 1), vza=vza[:, i].reshape(1, 1),
        #                                   vaa=vaa[:, i].reshape(1, 1), saa=saa[:, i].reshape(1, 1),
        #                                   taur=taur[:, i, :].reshape(1, 1, bands.size),
        #                                   La=La[:, i, :].reshape(1, 1, bands.size), F0=F0_t[0, i, :],
        #                                   windspeed=winds_peed[:, i].reshape(1, 1),
        #                                   winddirection=wd_rad[:, i].reshape(1, 1),
        #                                   taua=aot[:, i, :].reshape(1, 1, bands.size), iter_num=1, mode=2)

        # 9. 水体贡献
        nLw = rrs_w * F0_t
        csza = np.zeros(shape=(sza.shape[0], sza.shape[1], 1))
        csza[:, :, 0] = np.cos(sza * np.pi / 180)
        Lw = nLw * t_solar * tg_sol * csza / fsol.reshape(1, fsol.size, 1)
        tLw = Lw / tg_sol * t_sensor
        # 10. Ltoa
        Lt_simu = tLf + Lr + La + tLw

        out = np.full(shape=(Lt_simu.shape[1], Lt_simu.shape[2] + 5), fill_value=np.nan)
        out[:, 0:8] = Lt_simu[0, :, :]
        out[:, 8] = aot[0, :, 0]
        out[:, 9] = rrs_w[0, :, 0]
        out[:, 10] = sza
        out[:, 11] = vza
        out[:, 12] = reaa
        print()
        # 11. 导出计算结果
        df = pd.DataFrame(out, columns=[str(i) + "nm" for i in bands] + ["aot", "rrs_w", "sza", "vza", "reaa"],
                          index=None)

        # 在不同版本的pandas下，写出到excel命令可能会覆盖掉本来的数据表，因此，请备份原始数据
        # with pd.ExcelWriter(insitu_file, engine='openpyxl', mode='a') as writer:
        df.to_excel(insitu_file, sheet_name="Ltoa_simulation_" + sensorID, index=True)

        # try:
        #     # 在不同版本的pandas下，写出到excel命令可能会覆盖掉本来的数据表，因此，请备份原始数据
        #     with pd.ExcelWriter(insitu_file, engine='openpyxl', mode='a') as writer:
        #         df.to_excel(writer, sheet_name="Ltoa_simulation_" + sensorID, index=True)
        # except:
        #     # 无法导出到excel，则导出到csv，通常可能是缺少openxyl库
        #     df.to_csv(insitu_file + ".csv", index=True)
        return


if __name__ == '__main__':
    simula().run_main()
