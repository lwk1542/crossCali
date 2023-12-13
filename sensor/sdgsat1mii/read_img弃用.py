# -*- coding: utf-8 -*-
"""
@Time    : 2023/4/3 10:09
@FileName: read_img.py
@Project : git_repository
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""
from sharepy import read_tif
from sharepy import read_sensorinfo
import numpy as np
import os
from sharepy import reproject
from osgeo import gdal
from xml.etree import ElementTree as ET


class Read(object):
    def __init__(self, filepath:str,north:float =None,south:float =None,west:float =None,east:float =None,
                 resize:float=None):

        self.filepath = filepath
        self.north, self.south, self.west, self.east=north, south, west, east
        self.resize = resize

    def run_main(self):
        meta_dict = self.meta_data(filepath=self.filepath)
        img_dict = self.imagery_info()
        sensor_dict = self.sensor_info()
        sza = np.zeros_like(img_dict["lon"]) + float(meta_dict["sza"])
        saa = np.zeros_like(img_dict["lon"]) + float(meta_dict["saa"])
        vza = np.zeros_like(img_dict["lon"]) + 5
        vaa = np.zeros_like(img_dict["lon"]) + 100
        dict_out = {**meta_dict, **img_dict, **sensor_dict}
        dict_out["sza"] = sza
        dict_out["saa"] = saa
        dict_out["vza"] = vza
        dict_out["vaa"] = vaa

        return dict_out

    def imagery_info(self):
        # 合并，裁剪
        roitif = self.roi(filepath=self.filepath)
        infile_temp1 = reproject.convert_coordinates(roitif)
        ds = read_tif.GdalReadTif(in_file=infile_temp1)
        lon, lat = ds.get_lon_lat()
        lon[lon < 0] = lon[lon < 0] + 360
        dn = np.dstack(
            [ds.get_data(1), ds.get_data(2), ds.get_data(3), ds.get_data(4), ds.get_data(5), ds.get_data(6),
             ds.get_data(7)]
        )
        return {"lon": lon, "lat": lat, "dn": dn}

    def sensor_info(self):
        sensorinfo = read_sensorinfo.read_sensorinfo(sensorinfo_file="simulateRadiance/share/sdgsat1mii/msl12_sensor_info.dat")
        [wavelength, F0, Tau_r, k_oz, t_co2, k_no2, a_h2o, b_h2o, c_h2o, d_h2o, e_h2o, f_h2o, g_h2o, awhite, aw, bbw,
         oobwv, ooblw, wed, waph] = sensorinfo

        info = {"bands": np.array(wavelength), "F0": np.array(F0).reshape(-1, 1), "Tau_r": np.array(Tau_r), "k_oz": np.array(k_oz),
                "t_co2": np.array(t_co2), "k_no2": np.array(k_no2), "zia_table": [np.array(a_h2o), np.array(b_h2o),
                 np.array(c_h2o),  np.array(d_h2o),  np.array(e_h2o),  np.array(f_h2o), np.array(g_h2o)],
                "awhite": np.array(awhite), "aw": np.array(aw), "bbw": np.array(bbw),
                "oobwv": np.array(oobwv), "ooblw": np.array(ooblw), "wed": np.array(wed), "waph": np.array(waph)}
        return info

    def meta_data(self, filepath):
        calib_file_path = filepath + os.sep + os.path.basename(filepath) + ".calib.xml"
        meta_file_path = filepath + os.sep + os.path.basename(filepath) + ".meta.xml"
        tree = ET.parse(meta_file_path)
        time = tree.find("SatelliteInfo").find("CenterTime").find("Acamera").text
        saa = tree.find("SatelliteInfo").find("SolarAzimuth").text
        sza = tree.find("SatelliteInfo").find("SolarZenith").text
        roll = tree.find("SatelliteInfo").find("RollSatelliteAngle").text
        pitch = tree.find("SatelliteInfo").find("PitchSatelliteAngle").text
        yaw = tree.find("SatelliteInfo").find("YawSatelliteAngle").text
        meta_dict = {"time": time, "saa": saa, "sza": sza, "roll": roll, "pitch": pitch, "yaw": yaw}
        return meta_dict

    def roi(self, filepath):
        rangelonlat=[self.west,  self.north, self.east, self.south]
        outtif = filepath + os.sep + os.path.basename(filepath) + "_ROI.tif"
        if os.path.exists(outtif):
            if os.path.getsize(outtif) / float(1024) / float(1024) < 1:
                os.remove(outtif)
            else:
                return outtif
        if not os.path.exists(outtif):
            print("merge and clip..........")
            filesep = [filepath + os.sep + os.path.basename(filepath) + "_A.tif",
                       filepath + os.sep + os.path.basename(filepath) + "_B.tif"]
            outds = gdal.BuildVRT("", filesep, separate=False)
            dataset = gdal.Open(filesep[0])
            coords1 = reproject.lonlat2geo(dataset, rangelonlat[0], rangelonlat[1])
            coords2 = reproject.lonlat2geo(dataset, rangelonlat[2], rangelonlat[3])
            spatialwin = [coords1[0], coords1[1], coords2[0], coords2[1]]
            options_list = [
                '-outsize ' + str(self.resize * 100) + '% ' + str(self.resize * 100) + '%' +
                ' -r average'
            ]
            options_string = " ".join(options_list)
            outtif_ = filepath + os.sep + os.path.basename(filepath) + "_ROI_temp.tif"
            outds_ = gdal.Translate(outtif_, outds, projWin=spatialwin)
            outds_1 = gdal.Translate(outtif, outds_, options=options_string)
            outds_= None
            outds= None
            outds_1= None
            os.remove(outtif_)
        return outtif


def calib_xml(xmlfile):
    # xmlfile = "L4A.calib.xml"
    # f = open(file, "r", encoding="gb2312") # gb2312格式不好用
    # datasource = f.read()
    # per = ET.parse(datasource)
    gains = np.array([0.051560133, 0.036241353, 0.023316835, 0.015849666, 0.016096381, 0.019719039, 0.013811458])
    bias = np.array([0, 0, 0, 0, 0, 0, 0])
    return gains, bias


if __name__ == '__main__':
    outpath = r"F:\SDG\rrs"
    inpath = r"G:\SDGsat\pearlRiverEstuary\20230106\KX10_MII_20220311_E114.52_N23.00_202200103222_L4A"
    rangelonlat = [112.93, 22.77, 114.35, 21.93]
    a = Read(filepath=inpath, north=22.77, south=21.93, west=112.93, east=114.35, resize=0.01)
    a.run_main()




