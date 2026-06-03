# -*- coding: utf-8 -*-
"""
@Time    : 2023/12/21 14:54
@FileName: __init__.py
@Project : atmospheric_correction
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""
# gcp校正: seadas 校正结果
from osgeo import osr
from osgeo import gdal
import numpy as np
import h5py
import os


# 创建临时文件
def write_bands(im_data, banddes=None):
    # 判断栅格数据的数据类型
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    # 判读数组维数
    if len(im_data.shape) == 3:
        im_height, im_width, im_bands = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    # 创建文件
    # 数据类型必须有，因为要计算需要多大内存空间
    driver = gdal.GetDriverByName("MEM")
    dataset = driver.Create("", im_width, im_height, im_bands, datatype)

    # 写入数组数据
    if im_bands == 1:
        # dataset.GetRasterBand(1).SetNoDataValue(65535)
        try:
            dataset.GetRasterBand(1).WriteArray(im_data)  # 写入
        except:
            dataset.GetRasterBand(1).WriteArray(im_data[:, :, 0])
    else:
        # if banddes==None:
        # banddes = ['Rrs_412', 'Rrs_443', 'Rrs_490', 'Rrs_520', 'Rrs_565', 'Rrs_670', 'chlor_a']
        for i in range(im_bands):
            try:
                # dataset.GetRasterBand(i + 1).SetNoDataValue(65535)
                RasterBand = dataset.GetRasterBand(i + 1)
                # RasterBand.SetDescription(banddes[i])
                RasterBand.WriteArray(im_data[:, :, i])
            except IndentationError:
                print('band:' + i)
    return dataset


def gcp_main(outfile: str, bands_name: list[str], value: np.ndarray, latitude: np.ndarray,
             longitude: np.ndarray,resolution:float=0.01):
        # 将波段数据写入临时内存文件
        image: gdal.Dataset = write_bands(value)

        # 控制点列表, 设置7*7个控制点
        gcps = []
        x_arr = np.linspace(0, longitude.shape[1] - 1, num=7, endpoint=True, dtype=int)
        y_arr = np.linspace(0, longitude.shape[0] - 1, num=7, endpoint=True, dtype=int)
        for x in x_arr:
            for y in y_arr:
                if abs(longitude[y, x]) > 180 or abs(latitude[y, x]) > 90:
                    continue
                gcps.append(gdal.GCP(np.float64(longitude[y, x]), np.float64(latitude[y, x]),
                                     0, np.float64(x), np.float64(y)))
        # 设置空间参考
        sr = osr.SpatialReference()
        sr.ImportFromEPSG(4326)
        sr.MorphToESRI()

        # 给数据及设置控制点及空间参考
        image.SetGCPs(gcps, sr.ExportToWkt())
        # options = gdal.WarpOptions(dstSRS='EPSG:4326', cropToCutline=True)
        dst = gdal.Warp(outfile, image, format='GTiff', tps=True, dstNodata=np.nan,
                        resampleAlg=gdal.GRA_NearestNeighbour)  # dstNodata=65535

        for i, bandname in enumerate(bands_name):
            band = dst.GetRasterBand(i + 1)
            band.SetDescription(bandname)  # 设置波段描述（波段名）
        
        # 确保修改已保存
        dst.FlushCache()  # 将内存中的修改保存到磁盘

        image= None
        dst = None
        


def read_h5(h5, data_id):

    ds = h5[data_id]
    value = ds[()]*1.
    try:
        Fillvalue = ds.attrs["_FillValue"]
        value[value == Fillvalue[0]] = np.nan
    except:
        pass
    try:
        valid_max = ds.attrs["valid_max"]
        valid_min = ds.attrs["valid_min"]
        value[value < valid_min[0]] = np.nan
        value[value > valid_max[0]] = np.nan
    except:
        pass
    try:
        scale_factor = ds.attrs["scale_factor"]
        value = value * scale_factor[0]
    except:
        pass
    try:
        add_offset = ds.attrs["add_offset"]
        value = value + add_offset[0]
    except:
        pass
    value[value==0]=np.nan
    return value


def sdgsat1mii():
    path = r"G:\SDGsat\calibration\sea\2023\validation\turbid\test"
    file = path + os.sep + "KX10_MII_20230326_E123.53_N36.50_202300171003_L4B_ROI_L2.H5"
    h5 = h5py.File(file, mode="r")
    lat_tar = read_h5(h5=h5, data_id="navigation_data/latitude")
    lon_tar = read_h5(h5=h5, data_id="navigation_data/longitude")
    wavelength = [401, 438, 495, 553, 657, 776, 854]

    for i, b_ in enumerate(wavelength):
        _ = read_h5(h5=h5, data_id="geophysical_data/" + "rhos" + "_" + str(b_))
        if i == 0:
            (rows, columns) = _.shape
            value = np.zeros(shape=(rows, columns, wavelength.__len__())) + np.nan
        value[:, :, i] = _
    outfile = os.path.dirname(file)+os.sep+os.path.splitext(os.path.basename(file))[0]+"_rhos.tif"
    gcp_main(outfile=outfile, bands_name=[str(j) for j in wavelength], value=value, longitude=lon_tar, latitude=lat_tar)


def sentinel3olci(ifile: str, wavelength:list,outfile:str):
    # path = r"G:\SDGsat\calibration\sea\2023\validation\turbid\test"
    # file = path + os.sep + "S3A_OL_1_EFR____20230326T015501_20230326T015801_20230327T023511_0179_097_060_2340_PS1_O_NT_003_L2.H5"
    h5 = h5py.File(ifile, mode="r")
    lat_tar = read_h5(h5=h5, data_id="navigation_data/latitude")
    lon_tar = read_h5(h5=h5, data_id="navigation_data/longitude")
    # wavelength = [400, 443, 490, 560,  665, 779,  865]

    for i, b_ in enumerate(wavelength):
        _ = read_h5(h5=h5, data_id="geophysical_data/" + "rhos" + "_" + str(b_))
        if i == 0:
            (rows, columns) = _.shape
            value = np.zeros(shape=(rows, columns, wavelength.__len__())) + np.nan
        value[:, :, i] = _
    # outfile = os.path.dirname(ifile)+os.sep+os.path.splitext(os.path.basename(ifile))[0]+"_rhos.tif"
    gcp_main(outfile=outfile, bands_name=[str(j) for j in wavelength], value=value, longitude=lon_tar, latitude=lat_tar)


def run_seadas_file(ifile:str, geo_para:list[str], outfile:float):
    h5 = h5py.File(ifile, mode="r")
    lat_tar = read_h5(h5=h5, data_id="navigation_data/latitude")
    lon_tar = read_h5(h5=h5, data_id="navigation_data/longitude")
    # wavelength = [400, 443, 490, 560, 665, 779, 865]

    for i, b_ in enumerate(geo_para):
        _ = read_h5(h5=h5, data_id="geophysical_data/" + b_)
        if i == 0:
            (rows, columns) = _.shape
            value = np.zeros(shape=(rows, columns, geo_para.__len__())) + np.nan
        value[:, :, i] = _
    # outfile = os.path.dirname(ifile) + os.sep + os.path.splitext(os.path.basename(ifile))[0] +"_"+b_.split("_")[0]+".tif"
    gcp_main(outfile=outfile, bands_name=geo_para, value=value, longitude=lon_tar,
             latitude=lat_tar)


def parameter(sensorid: str, para: list[str] = ["rhom", "rhos"]):
    bands = {"landsat8oli": [443, 482, 561, 655, 865]}
    return [p + "_" + str(j) for p in para for j in bands[sensorid]]


def landsat():
    import fileSearch
    path = r"G:\test\landsat\LC08_L1TP_121040_20231017_20231103_02_T1"
    files = fileSearch.get_filelist(idx=["LANDSAT8_OLI", ".L2.OC.nc"], path=path, mode="all")
    para = parameter(sensorid="landsat8oli", para=["rhom"])
    res=0.001
    for file in files:
        run_seadas_file(file=file, geo_para=para, res=res)


def oceancolor_l3():
    from netCDF4 import Dataset
     # 打开.nc文件
    nc_file = r"E:\XDA/AQUA_MODIS.20020701_20210731.L3m.MC.SST.sst.4km.nc"
    nc_dataset = Dataset(nc_file, "r")

    # 获取文件中的变量
    variables = nc_dataset.variables

    # 获取每个变量的属性和维度
    lat = variables["lat"]
    lon = variables["lon"]
    lon_tar, lat_tar = np.meshgrid(lon, lat)
    value = variables["sst"][()]
    value[value<0]=np.nan
    value=value.reshape(value.shape[0],value.shape[1],1)
    # 范围：
    loc1=np.where((lat_tar.data<35)&(lat_tar.data>-5)&(lon_tar.data<140)&(lon_tar.data>100))
    up, low, left, right = np.min(loc1[0]), np.max(loc1[0]), np.min(loc1[1]), np.max(loc1[1])

    outfile = os.path.dirname(nc_file)+os.sep+os.path.basename(nc_file)[0:-3]+"_roi.tif"
    gcp_main(outfile=outfile, value=value[up:low, left:right], longitude=lon_tar.data[up:low, left:right],bands_name=["sst"],
             latitude=lat_tar.data[up:low, left:right], resolution=0.01)
    #
    # for var_name in variables:
    #     var = variables[var_name]
    #     print("Variable:", var_name)
    #     print("Shape:", var.shape)
    #     print("Dimensions:", var.dimensions)
    #     print("Attributes:")
    #     for attr_name in var.ncattrs():
    #         print(f"\t{attr_name}: {getattr(var, attr_name)}")

    # 关闭文件
    nc_dataset.close()

if __name__ == '__main__':
    print('用于各种数据转geotiff格式')
    # sdgsat1mii()
    # sentinel3olci()
    # landsat()
    oceancolor_l3()