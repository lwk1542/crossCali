# -*- coding: utf-8 -*-
"""
@Time    : 2023/4/3 11:06
@FileName: read_sensorinfo.py
@Project : git_repository
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""
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