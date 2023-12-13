# -*- coding: utf-8 -*-
"""
@Time    : 2023/9/28 14:42
@FileName: input.py
@Project : atmospheric_correction
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
主函数
"""
import os
import pandas as pd
import numpy as np
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


def commom_variables():
    hico_dir = r"D:\researchProject_lwk\atmospheric_correction\oceancolor_acnirv2\share\hico"
    base_dir = r'D:\researchProject_lwk\atmospheric_correction\oceancolor_acnirv2'
    rsr_dir = base_dir + os.sep + "RSR"
    rsr_infile = rsr_dir + os.sep + 'RSR.xlsx'
    infofile_taur = rsr_dir + os.sep + "taur.txt"
    infofile_ozone = rsr_dir + os.sep + 'Ozoneattenuationcoefficients'
    infofile_no2 = rsr_dir + os.sep + 'NO2absorption'
    sensorid = "fengyun3dmersi"  # "goci"
    lut_target_dir = base_dir + os.sep + "share"+os.sep+sensorid

    F0, center_wave = f0(rsr_infile=rsr_infile, sensorid=sensorid)
    Nbands = center_wave.size
    print(F0, center_wave)
    # F0 = [1732.008, 1890.81, 1967.733, 1833.29, 1518.74, 1474.612, 1277.482, 953.952]
    return rsr_infile, lut_target_dir, hico_dir, infofile_taur, infofile_ozone, infofile_no2, Nbands, center_wave, F0,sensorid


def f0(rsr_infile, sensorid):
    thuillier_F0_file = r'D:\researchProject_lwk\atmospheric_correction\oceancolor_acnirv2\RSR/Thuillier_F0.txt'
    f_reference = open(thuillier_F0_file, "r")
    lines = f_reference.readlines()[:]
    wavelength = [float(i.split(" ", -1)[0]) for i in lines]
    value = [float(i.split(" ", -1)[1]) for i in lines]
    f0_df = pd.DataFrame({"wavelength": wavelength,
                          "value": value
                         })
    target_rsr = pd.read_excel(io=rsr_infile, sheet_name=sensorid, header=0, index_col=None)
    f0 = calculate_band_average(reference_spectrum=f0_df, spectrum_response_function=target_rsr)
    # center_wave = target_rsr.columns.values[0:]
    center_wave = target_rsr.columns[1::2]
    return f0, center_wave


def main():

    rsr_infile, lut_target_dir, hico_dir, infofile_taur, infofile_ozone, infofile_no2, Nbands, centr_wave, F0, sensorid = commom_variables()
    if not os.path.exists(lut_target_dir):
        os.mkdir(lut_target_dir)
    infofile_target = lut_target_dir + os.sep + "msl12_sensor_info.dat"
    import sensorInfo_generating
    sensorInfo_generating.SensorInfo(rsr_infile=rsr_infile, infofile_target=infofile_target,
                                     infofile_hico=hico_dir + os.sep + "msl12_sensor_info.dat",
                                     infofile_taur=infofile_taur, infofile_ozone=infofile_ozone,
                                     infofile_no2=infofile_no2, Nbands=Nbands, centr_wave=centr_wave, F0=F0,
                                     sensorid=sensorid).run_main()

    import rayleigh_simulated_hico
    out_path_ray = lut_target_dir+os.sep+"rayleigh"
    if not os.path.exists(out_path_ray):
        os.mkdir(out_path_ray)
    rayleigh_simulated_hico.run_main(path_rsr=rsr_infile,
                                     path_hico=hico_dir+os.sep+"rayleigh",
                                     out_path=out_path_ray,
                                     sensorid=sensorid)

    import aerosol_simulated_hico
    out_path_aer = lut_target_dir + os.sep + "aerosol"
    if not os.path.exists(out_path_aer):
        os.mkdir(out_path_aer)
    aerosol_simulated_hico.run_main(rsr_path=rsr_infile,
                                    path_hico=hico_dir+os.sep+"aerosol",
                                    path_out=out_path_aer,
                                    sensorid=sensorid)


if __name__ == '__main__':
    main()



