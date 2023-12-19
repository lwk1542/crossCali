# -*- coding: utf-8 -*-
"""
@Time    : 2022/11/12 15:03
@FileName: read_lut.py
@Project : git_repository
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
@ 使用分块计算后，每次单独加载查找表会增加时间消耗，因此，在识别传感器后，首先加载瑞利和气溶胶查找表以及传感器的其它信息
"""
import numpy as np
from netCDF4 import Dataset
import os

try:
    from pyhdf.SD import SD, SDC
except:
    pass


class CommonVariable(object):
    def __init__(self, sensor_id: str = None):
        self.bands = None
        self.aerosol_lut_path = None
        self.rayleigh_lut_path = None
        self.lut_path = None
        self.sensor_id = sensor_id

    def get(self):
        self.lut_path = self.get_lookup_table()
        self.rayleigh_lut_path = self.lut_path + os.sep + "rayleigh"
        self.aerosol_lut_path = self.lut_path + os.sep + "aerosol"
        # 2传感器信息
        sensor_info = self.read_sensor_info()
        self.bands = sensor_info[0]
        rayleigh_lut_info = self.read_rayleigh_lut()
        aerosol_lut_info = self.read_aerosol_lut()
        print("successfully loaded sensor info and look-up table...")
        return sensor_info, rayleigh_lut_info,  aerosol_lut_info
        # return sensor_info, self.rayleigh_lut_path, self.aerosol_lut_path

    def get_lookup_table(self):
        # 根据指定的传感器获取查找表路径
        print('sensorID: ' + self.sensor_id)
        match self.sensor_id:
            case 'hy1ccocts':
                lut_path = r"../share" + os.sep + 'hy1ccocts'
            case 'hy1dcocts':
                lut_path = r"../share" + os.sep + 'hy1dcocts'
            case 'fy3dmersi':
                lut_path = r"../share" + os.sep + "fy3dmersi"
            case "sdgsat1mii":
                lut_path = r"../share" + os.sep + "sdgsat1mii"
            case 'seawifsphd':
                lut_path = r"../share" + os.sep + "seawifsphd"
            case "landsat8oli":
                lut_path = r"../share" + os.sep + "landsat8oli"
            case _:
                print("Error: Can not identify satellite sensor ID for obtaining look-up table... ")
        # if self.sensor_id == 'hy1acocts':
        #     lut_path = r"../share" + os.sep + "cocts" + os.sep + "hy1a"
        # elif self.sensor_id == 'hy1bcocts':
        #     lut_path = r"../share" + os.sep + "cocts" + os.sep + "hy1b"
        # elif self.sensor_id == 'hy1ccocts':
        #     lut_path = r"../share" + os.sep + 'hy1ccocts'
        # elif self.sensor_id == 'hy1dcocts':
        #     lut_path = r"../share" + os.sep + 'hy1dcocts'
        # elif self.sensor_id == 'fy3dmersi':
        #     lut_path = r"../share" + os.sep + "fy3dmersi"
        # elif self.sensor_id == 'olcis3a':
        #     lut_path = r"../share" + os.sep + "olci" + os.sep + "s3a"
        # elif self.sensor_id == 'olcis3b':
        #     lut_path = r"../share" + os.sep + "olci" + os.sep + "s3b"
        # elif self.sensor_id == 'l8oli':
        #     lut_path = r"../share" + os.sep + "oli"
        # elif self.sensor_id == 'modist':
        #     lut_path = r"../share" + os.sep + "modis" + os.sep + "terra"
        # elif self.sensor_id == 'modisa':
        #     lut_path = r"../share" + os.sep + "modis" + os.sep + "aqua"
        # elif self.sensor_id == "sdgsat1mii":
        #     lut_path = r"../share" + os.sep + "sdgsat1mii"
        # elif self.sensor_id == 'seawifsphd':
        #     lut_path = r"../share" + os.sep + "seawifsphd"
        # else:
        #     print("Error: Can not identify satellite sensor ID for obtaining look-up table... ")
        return lut_path

    def read_sensor_info(self):
        """
        Args:
            sensorinfo_file (): msl12_sensor_info.dat文件，有标准格式
        Returns:
            传感器参数
        """
        f = open(self.lut_path+os.sep+"msl12_sensor_info.dat")
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
        Zia_tabel = [np.array(a_h2o), np.array(b_h2o), np.array(c_h2o), np.array(d_h2o), np.array(e_h2o),
                     np.array(f_h2o), np.array(g_h2o)]
        return [np.array(wavelength), np.array(F0), np.array(Tau_r), np.array(k_oz), np.array(t_co2), np.array(k_no2),
               Zia_tabel, np.array(awhite), np.array(aw), np.array(bbw), np.array(oobwv), np.array(ooblw),
               np.array(wed), np.array(waph)]

    def read_rayleigh_lut(self):
        taur = []
        i_ray = []
        for i, band in enumerate(self.bands):
            rayleigh_lut = self.rayleigh_lut_path + os.sep + 'rayleigh_' + self.sensor_id + "_" + str(int(band)) + '_iqu.hdf'
            rayDtset = Dataset(rayleigh_lut)
            taur.append(rayDtset.variables['taur'][:])
            i_ray.append(rayDtset.variables['i_ray'][:])
            # q_ray = rayDtset.variables['q_ray'][:]
            # u_ray = rayDtset.variables['u_ray'][:]
            if i == 0:
                # depol = rayDtset.variables['depol'][:]
                senz = rayDtset.variables['senz'][:]
                solz = rayDtset.variables['solz'][:]
                try:
                    sigma = rayDtset.variables['sigma'][:]
                except:
                    wind_speed = rayDtset.variables['wind'][:]
                    sigma = np.sqrt(0.00534 * wind_speed)

        data_dict = {}
        data_dict["taur"] = np.array(taur).reshape(-1)
        data_dict["senz"] = np.array(senz)
        data_dict["solz"] = np.array(solz)
        data_dict["sigma"] = np.array(sigma)
        data_dict["i_ray"] = np.array(i_ray)
        return data_dict

    def read_aerosol_lut(self):
        # loads the entire aerosol model table for the specified model list
        # 把全部80个模型全部读入内存，以便之后快速筛选

        models = ['r30f95', 'r30f80', 'r30f50', 'r30f30', 'r30f20', 'r30f10', 'r30f05', 'r30f02',
                  'r30f01', 'r30f00', 'r50f95', 'r50f80', 'r50f50', 'r50f30', 'r50f20', 'r50f10',
                  'r50f05', 'r50f02', 'r50f01', 'r50f00', 'r70f95', 'r70f80', 'r70f50', 'r70f30',
                  'r70f20', 'r70f10', 'r70f05', 'r70f02', 'r70f01', 'r70f00', 'r75f95', 'r75f80',
                  'r75f50', 'r75f30', 'r75f20', 'r75f10', 'r75f05', 'r75f02', 'r75f01', 'r75f00',
                  'r80f95', 'r80f80', 'r80f50', 'r80f30', 'r80f20', 'r80f10', 'r80f05', 'r80f02',
                  'r80f01', 'r80f00', 'r85f95', 'r85f80', 'r85f50', 'r85f30', 'r85f20', 'r85f10',
                  'r85f05', 'r85f02', 'r85f01', 'r85f00', 'r90f95', 'r90f80', 'r90f50', 'r90f30',
                  'r90f20', 'r90f10', 'r90f05', 'r90f02', 'r90f01', 'r90f00', 'r95f95', 'r95f80',
                  'r95f50', 'r95f30', 'r95f20', 'r95f10', 'r95f05', 'r95f02', 'r95f01', 'r95f00']

        aerosol_models = {}
        for i in range(models.__len__()):
            aerofile = self.aerosol_lut_path + os.sep + "aerosol_" + self.sensor_id + "_" + models[i] + "v01.hdf"
            aerosol_model = {'name': i}
            aero_paras = ['wave', 'scatt', 'albedo', 'extc', 'angstrom', 'phase', 'solz', 'senz', 'phi', 'acost',
                          'bcost', 'ccost', 'dtran_wave', 'dtran_theta', 'dtran_a', 'dtran_b', 'dtran_a0', 'dtran_b0']
            try:
                f = SD(aerofile, SDC.READ)
                for aero_para in aero_paras:
                    if aero_para == 'phase':
                        aerosol_model[aero_para + '_lut'] = np.log((f.select(aero_para)).get())
                    else:
                        aerosol_model[aero_para + '_lut'] = (f.select(aero_para)).get()
            except:
                f = Dataset(aerofile, mode='r')
                for aero_para in aero_paras:
                    if aero_para == 'phase':
                        aerosol_model[aero_para + '_lut'] = np.log(f.variables[aero_para][()])
                    else:
                        aerosol_model[aero_para + '_lut'] = f.variables[aero_para][()]
            aerosol_models[models[i]] = aerosol_model

        return aerosol_models
