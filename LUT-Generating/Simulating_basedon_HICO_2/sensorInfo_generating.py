# -*- coding: utf-8 -*-
"""
@Time    : 2022/11/3 14:43
@FileName: sensorInfo_generating.py
@Project : git_repository
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""
import os
import numpy as np
import pandas as pd
import scipy.interpolate as interpolate


def interpolate_fun(x, y, x_new=None):
    """
    插值函数
    """
    if type(x) is np.ndarray:
        x = x
    else:
        # dataframe
        x = x.values
    if type(y) is np.ndarray:
        y = y
    else:
        y = y.values

    f = interpolate.interp1d(x, y, kind="linear", bounds_error=False)
    y_new = f(x_new)
    x_new = x_new
    return [x_new, y_new]


def calculate_band_average(reference_spectrum=None, spectrum_response_function=None):
    """
    插值
    积分
    """
    reference_wave = reference_spectrum.iloc[:, 0]
    reference_response = reference_spectrum.iloc[:, 1]
    # 波段数：
    bands = int(spectrum_response_function.shape[1]/2)
    band_values = []
    for i in range(bands):
        srf_wave = spectrum_response_function.iloc[:, 2*i]
        srf_band = spectrum_response_function.iloc[:, 2*i+1]
        # high_boundary = np.min(np.nanmax(reference_wave), np.nanmax(srf_wave))
        # low_boundary = np.max(np.nanmin(reference_wave), np.nanmin(srf_wave))
        wave, solar_spectrum_new = interpolate_fun(reference_wave, reference_response, x_new=srf_wave)
        band_values_ = np.nansum(solar_spectrum_new * srf_band) / np.nansum(srf_band)
        band_values.append(band_values_)

    return band_values


class SensorInfo(object):
    def __init__(self,rsr_infile, infofile_target, infofile_hico, infofile_taur, infofile_ozone, infofile_no2, Nbands, centr_wave, F0, sensorid):
        # 光谱响应函数
        # self.rsr_infile = r'D:\researchProject_lwk\atmospheric_correction\oceancolor_acnirv2\RSR\RSR.xlsx'
        # self.infofile_target = r"D:\researchProject_lwk\atmospheric_correction\oceancolor_acnirv2\share\goci" + os.sep + "msl12_sensor_info.dat"
        # self.infofile_hico = r"D:\researchProject_lwk\atmospheric_correction\oceancolor_acnirv2\share\hico" + os.sep + "msl12_sensor_info.dat"
        # self.infofile_taur = "taur.txt"
        # self.infofile_ozone = r'D:\researchProject_lwk\atmospheric_correction\oceancolor_acnirv2\RSR/Ozoneattenuationcoefficients'
        # self.infofile_no2 = r'D:\researchProject_lwk\atmospheric_correction\oceancolor_acnirv2\RSR/NO2absorption'
        # self.Nbands = 8
        # self.centr_wave = [412, 443, 490, 555, 660, 680, 745, 865]
        # self.F0 = [1732.008, 1890.81, 1967.733, 1833.29, 1518.74, 1474.612, 1277.482, 953.952]
        self.rsr_infile = rsr_infile
        self.infofile_target = infofile_target
        self.infofile_hico = infofile_hico
        self.infofile_taur = infofile_taur
        self.infofile_ozone = infofile_ozone
        self.infofile_no2 = infofile_no2
        self.Nbands = Nbands
        self.centr_wave = centr_wave
        self.F0 = F0
        self.sensorid=sensorid

    def run_main(self):
        self.write_target()

    def target_rsr(self):
        rsr = pd.read_excel(io=self.rsr_infile, sheet_name=self.sensorid, header=0, index_col=None)
        # wavelength=rsr['Wavelength (nm)']
        # rsr = rsr[(rsr['Wavelength (nm)'] >= 370) & (rsr['Wavelength (nm)'] <= 980)]
        return rsr

    def read_hico(self):
        f_reference = open(self.infofile_hico, "r")
        lines = f_reference.readlines()
        # filter note
        lines = [i for i in lines if not "#" in i.lower()]
        wavelength = [float(i.split("= ", 1)[-1].split()[0]) for i in lines if "lambda(" in i.lower()]
        aw = [float(i.split("= ", 1)[-1].split()[0]) for i in lines if "aw(" in i.lower()]
        bbw = [float(i.split("= ", 1)[-1].split()[0]) for i in lines if "bbw(" in i.lower()]
        aw_ = pd.DataFrame({"wavelength": wavelength,
                            "value": aw
                            })
        bbw_ = pd.DataFrame({"wavelength": wavelength,
                             "value": bbw
                             })
        return aw_, bbw_

    def read_taur(self):
        f_reference = open(self.infofile_taur, "r")
        lines = f_reference.readlines()[15:]
        wavelength = [float(i.split(" ", -1)[0]) for i in lines]
        tau_r = [float(i.split(" ", -1)[1]) for i in lines]
        dpol = [float(i.split(" ", -1)[2]) for i in lines]
        Taur = pd.DataFrame({"wavelength": wavelength,
                             "value": tau_r
                             })
        Dpol = pd.DataFrame({"wavelength": wavelength,
                             "value": dpol
                             })
        return Taur, Dpol

    def read_koz(self):
        # 2. 臭氧衰减率
        o3 = pd.read_csv(self.infofile_ozone, index_col=None, header=None, sep=" ", skiprows=19)
        return o3

    def read_kno2(self):
        # no2衰减率
        no2 = pd.read_csv(self.infofile_no2, index_col=None, header=None, sep=" ", skiprows=19)
        return no2

    def write_target(self):
        f_target = open(self.infofile_target, "w")
        lines = ["# -----------------------------------------------------------",
                 "#"+self.sensorid+" sensor-specific atmospheric correction data",
                 "# -----------------------------------------------------------",
                 "", "#", "# Number of bands", "#", "Nbands = " + str(self.Nbands), "", "#", "# Wavelengths (um)", "#"]

        for i, line in enumerate(lines):
            f_target.write(line)
            f_target.write('\n')
        # lambda
        for i, line in enumerate(self.centr_wave):
            f_target.write("Lambda({0}) = {1}".format(i, line))
            f_target.write('\n')
        f_target.write(' ')
        f_target.write('\n')

        # F0
        for i, line in enumerate(self.F0):
            f_target.write("F0({0}) = {1}".format(i, line))
            f_target.write('\n')
        f_target.write(' ')
        f_target.write('\n')

        target_rsr = self.target_rsr()

        # Tau_r
        taur_hyper, dpol_hyper = self.read_taur()
        taur_bands = calculate_band_average(reference_spectrum=taur_hyper, spectrum_response_function=target_rsr)
        for i, line in enumerate(taur_bands):
            f_target.write("Tau_r({0}) = {1}".format(i, line))
            f_target.write('\n')
        f_target.write(' ')
        f_target.write('\n')

        # k_oz
        koz_hyper = self.read_koz()
        koz_bands = calculate_band_average(reference_spectrum=koz_hyper, spectrum_response_function=target_rsr)
        for i, line in enumerate(koz_bands):
            f_target.write("k_oz({0}) = {1}".format(i, line))
            f_target.write('\n')
        f_target.write(' ')
        f_target.write('\n')

        # k_no2
        kno2_hyper = self.read_kno2()
        kno2_bands = calculate_band_average(reference_spectrum=kno2_hyper, spectrum_response_function=target_rsr)
        for i, line in enumerate(kno2_bands):
            f_target.write("k_no2({0}) = {1}".format(i, line))
            f_target.write('\n')
        f_target.write(' ')
        f_target.write('\n')

        #  从HICO里面卷积的有两个参数：aw, bbw
        aw_hico, bbw_hico = self.read_hico()
        aw_bands = calculate_band_average(reference_spectrum=aw_hico, spectrum_response_function=target_rsr)
        for i, line in enumerate(aw_bands):
            f_target.write("aw({0}) = {1}".format(i, line))
            f_target.write('\n')
        f_target.write(' ')
        f_target.write('\n')

        bbw_bands = calculate_band_average(reference_spectrum=bbw_hico, spectrum_response_function=target_rsr)
        for i, line in enumerate(bbw_bands):
            f_target.write("bbw({0}) = {1}".format(i, line))
            f_target.write('\n')
        f_target.write(' ')
        f_target.write('\n')

        # CO2
        lines = ["#", "# co2 transmittance", "# Z. Amhad, May 2006", "#"]  # 近红外波段全为1
        for i, line in enumerate(lines):
            f_target.write(line)
            f_target.write('\n')
        for i, line in enumerate(self.centr_wave):
            f_target.write("t_co2({0}) = {1}".format(i, 1))
            f_target.write('\n')
        f_target.write(' ')
        f_target.write('\n')

        # 水汽
        lines = ["#", "# h2o absorption transittance function", "# Z. Amhad, May 2006", "#"]
        for i, line in enumerate(lines):
            f_target.write(line)
            f_target.write('\n')
        for i, line in enumerate(self.centr_wave):
            f_target.write("a_h2o({0}) = {1}".format(i, 1.0))
            f_target.write('\n')
        f_target.write(' ')
        f_target.write('\n')

        for i, line in enumerate(self.centr_wave):
            f_target.write("b_h2o({0}) = {1}".format(i, 0.0))
            f_target.write('\n')
        f_target.write(' ')
        f_target.write('\n')

        for i, line in enumerate(self.centr_wave):
            f_target.write("c_h2o({0}) = {1}".format(i, 0.0))
            f_target.write('\n')
        f_target.write(' ')
        f_target.write('\n')

        for i, line in enumerate(self.centr_wave):
            f_target.write("d_h2o({0}) = {1}".format(i, 0.0))
            f_target.write('\n')
        f_target.write(' ')
        f_target.write('\n')

        for i, line in enumerate(self.centr_wave):
            f_target.write("e_h2o({0}) = {1}".format(i, 0.0))
            f_target.write('\n')
        f_target.write(' ')
        f_target.write('\n')

        for i, line in enumerate(self.centr_wave):
            f_target.write("f_h2o({0}) = {1}".format(i, 0.0))
            f_target.write('\n')
        f_target.write(' ')
        f_target.write('\n')

        for i, line in enumerate(self.centr_wave):
            f_target.write("g_h2o({0}) = {1}".format(i, 0.0))
            f_target.write('\n')
        f_target.write(' ')
        f_target.write('\n')

        # White-cap albedo
        lines = ["#", "# White-cap albedo (Frouin, May 1999, interpolated)", "#"]
        for i, line in enumerate(lines):
            f_target.write(line)
            f_target.write('\n')
        wavelength = np.array([412, 443, 490, 510, 555, 670, 765, 865])
        a_wc = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.889, 0.760, 0.645])
        a_wc_inter = interpolate.interp1d(wavelength, a_wc, bounds_error=False, fill_value="extrapolate",
                                          kind="linear")(self.centr_wave.astype(float))
        for i, line in enumerate(a_wc_inter):
            f_target.write("awhite({0}) = {1}".format(i, line))
            f_target.write('\n')
        f_target.write(' ')
        f_target.write('\n')

        lines = ['#', '# Out-of-band water-vapor functions (Yang algorithm)', '# from setting', '#', '# NEEDS UPDATE ',
                 '#']
        for i, line in enumerate(lines):
            f_target.write(line)
            f_target.write('\n')
        for i, line in enumerate(lines):
            f_target.write("oobwv01({0}) = 1.0".format(i))
            f_target.write('\n')
            f_target.write("oobwv02({0}) = 0.0".format(i))
            f_target.write('\n')
            f_target.write("oobwv03({0}) = 0.0".format(i))
            f_target.write('\n')
            f_target.write("oobwv04({0}) = 0.0".format(i))
            f_target.write('\n')
            f_target.write("oobwv05({0}) = 0.0".format(i))
            f_target.write('\n')
            f_target.write("oobwv06({0}) = 0.0".format(i))
            f_target.write('\n')
            f_target.write("oobwv07({0}) = 0.0".format(i))
            f_target.write('\n')
            f_target.write("oobwv08({0}) = 0.0".format(i))
            f_target.write('\n')
            f_target.write("oobwv09({0}) = 0.0".format(i))
            f_target.write('\n')
            f_target.write("oobwv10({0}) = 0.0".format(i))
            f_target.write('\n')
            f_target.write("oobwv11({0}) = 0.0".format(i))
            f_target.write('\n')
            f_target.write("oobwv12({0}) = 0.0".format(i))
            f_target.write('\n')
            f_target.write(' ')
            f_target.write('\n')

        f_target.close()


if __name__ == '__main__':
    SensorInfo().run_main()
