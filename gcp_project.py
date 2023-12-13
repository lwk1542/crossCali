# gcp校正: MODIS seadas 校正结果
from tqdm import tqdm
from osgeo import osr
from osgeo import gdal
import numpy as np
import h5py
import glob
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


class oct:
    @staticmethod
    def l2agcp(infile=None,
               radi_bands=['Rrs412', 'Rrs443', 'Rrs490', 'Rrs520', 'Rrs565', 'Rrs670', 'Rrs750', 'Rrs865', 'chlor_a'],
               latlon=False):
        f = h5py.File(infile, "r")
        longitude = f['/Navigation Data/Longitude'][()]
        latitude = f['/Navigation Data/Latitude'][()]
        # bands = ['Rrs412', 'Rrs443', 'Rrs490', 'Rrs520', 'Rrs565', 'Rrs670', 'chlor_a']

        if latlon:
            value = np.empty((longitude.shape[0], longitude.shape[1], len(radi_bands) + 2))
        else:
            value = np.empty((longitude.shape[0], longitude.shape[1], len(radi_bands)))

        for i, band in enumerate(radi_bands):
            dataset_band = f['/Geophysical Data/' + radi_bands[i]]
            value_band = dataset_band[()]
            value_band[value_band == -32767.] = np.nan
            value_band[value_band < -32767.] = np.nan
            value_band[value_band > 32767.] = np.nan
            # value_band[value_band <= 0] = np.nan
            value[:, :, i] = value_band
        f.close()

        # 将波段数据写入临时内存文件
        image: gdal.Dataset = write_bands(value)

        # 控制点列表, 设置7*7个控制点
        gcps = []
        x_arr = np.linspace(0, longitude.shape[1] - 1, num=7, endpoint=True, dtype=np.int)
        y_arr = np.linspace(0, longitude.shape[0] - 1, num=7, endpoint=True, dtype=np.int)
        for x in x_arr:
            for y in y_arr:
                if abs(longitude[y, x]) > 180 or abs(latitude[y, x]) > 90:
                    continue
                gcps.append(gdal.GCP(np.float64(longitude[y, x]), np.float64(latitude[y, x]),
                                     0,
                                     np.float64(x), np.float64(y)))

        # 设置空间参考
        sr = osr.SpatialReference()
        sr.ImportFromEPSG(4326)
        sr.MorphToESRI()

        # 给数据及设置控制点及空间参考
        image.SetGCPs(gcps, sr.ExportToWkt())
        if radi_bands[0][0:3] == 'Rrs':
            outfile = os.path.dirname(infile) + os.sep + os.path.basename(infile)[0:-3] + '-Rrs.geotiff'
        elif radi_bands[0][0:3] == 'Rrc':
            outfile = os.path.dirname(infile) + os.sep + os.path.basename(infile)[0:-3] + '-Rrc.geotiff'
        elif radi_bands[0] == 'chlor_a':
            outfile = os.path.dirname(infile) + os.sep + os.path.basename(infile)[0:-3] + '-chlor_a.geotiff'
        else:
            outfile = os.path.dirname(infile) + os.sep + os.path.basename(infile)[0:-3] + '_l2gen.geotiff'

        # 校正
        if latlon:
            cutlinelayer = radi_bands + ['latitude', 'longitude']
        else:
            cutlinelayer = radi_bands
        dst = gdal.Warp(outfile, image, format='GTiff', tps=True, xRes=0.01, yRes=0.01, dstNodata=np.nan,
                        resampleAlg=gdal.GRA_NearestNeighbour)  # dstNodata=65535

        for i, bandname in enumerate(cutlinelayer):
            band = dst.GetRasterBand(i + 1)
            band.SetMetadata({'bandname': bandname})
            band.SetDescription(bandname)

        image: None
        return outfile

    @staticmethod
    def l1agcp(infile=None, bands=None):
        f = h5py.File(infile, "r")
        Longitude = f['/Navigation Data/Longitude'][()]
        Latitude = f['/Navigation Data/Latitude'][()]
        bands = ['DN_412', 'DN_443', 'DN_490', 'DN_520', 'DN_565', 'DN_670', 'DN_750', 'DN_865']
        navi = ['Solar Zenith Angle', 'Solar Azimuth Angle', 'Satellite Zenith Angle', 'Satellite Azimuth Angle']
        value = np.empty((Longitude.shape[0], Longitude.shape[1], len(bands) + len(navi)))
        for i, band in enumerate(bands):
            dataset_band = f['/Geophysical Data/' + bands[i]]
            value_band = dataset_band[:, :] * 1.
            value_band[value_band == -32767.] = np.nan
            value[:, :, i] = value_band
        for i, band in enumerate(navi):
            dataset_band = f['/Navigation Data/' + navi[i]]
            value_band = dataset_band[:, :] * 1.
            value_band[value_band == -32767.] = np.nan
            value[:, :, i + len(bands)] = value_band

        f.close()

        # 将波段数据写入临时内存文件
        banddes = bands + navi
        image: gdal.Dataset = write_bands(value, banddes=banddes)

        # 控制点列表, 设置7*7个控制点
        gcps = []
        x_arr = np.linspace(0, Longitude.shape[1] - 1, num=7, endpoint=True, dtype=np.int)
        y_arr = np.linspace(0, Longitude.shape[0] - 1, num=7, endpoint=True, dtype=np.int)
        for x in x_arr:
            for y in y_arr:
                if abs(Longitude[y, x]) > 180 or abs(Latitude[y, x]) > 90:
                    continue
                gcps.append(gdal.GCP(np.float64(Longitude[y, x]), np.float64(Latitude[y, x]),
                                     0,
                                     np.float64(x), np.float64(y)))

        # 设置空间参考
        sr = osr.SpatialReference()
        sr.ImportFromEPSG(4326)
        sr.MorphToESRI()

        # 给数据及设置控制点及空间参考
        image.SetGCPs(gcps, sr.ExportToWkt())

        tempfile = os.path.dirname(infile) + os.sep + os.path.basename(infile)[0:-3] + '_temp.geotiff'

        cutlinelayer = bands + navi
        # 校正
        dst = gdal.Warp(tempfile, image, format='GTiff', tps=True, xRes=0.01, yRes=0.01, dstNodata=np.nan,
                        resampleAlg=gdal.GRA_NearestNeighbour, cutlineLayer=cutlinelayer)  # dstNodata=65535

        dst=None
        dst=gdal.Open(tempfile)
        spatialwin = [117., 41., 128., 20.]
        outfile = os.path.dirname(infile) + os.sep + os.path.basename(infile)[0:-3] + '_ROI.tiff'
        outds = gdal.Translate(outfile, dst, projWin=spatialwin)
        dst=None
        for i, bandname in enumerate(cutlinelayer):
            band =outds.GetRasterBand(i + 1)
            band.SetMetadata({'bandname': bandname})
            band.SetDescription(bandname)

        image= None
        outds=None
        return outfile

    @staticmethod
    def l1bgcp(infile=None, bands=None):
        f = h5py.File(infile, "r")
        Longitude = f['/Navigation Data/Longitude'][:, :]
        Latitude = f['/Navigation Data/Latitude'][:, :]
        bands = ['L_412', 'L_443', 'L_490', 'L_520', 'L_565', 'L_670', 'L_750', 'L_865']
        navi = ['Solar Zenith Angle', 'Solar Azimuth Angle', 'Satellite Zenith Angle', 'Satellite Azimuth Angle']
        value = np.empty((Longitude.shape[0], Longitude.shape[1], len(bands) + len(navi)))
        for i, band in enumerate(bands):
            dataset_band = f['/Geophysical Data/' + bands[i]]
            value_band = dataset_band[:, :] * 1.
            value_band[value_band == -32767.] = np.nan
            value[:, :, i] = value_band
        for i, band in enumerate(navi):
            dataset_band = f['/Navigation Data/' + navi[i]]
            value_band = dataset_band[:, :] * 1.
            value_band[value_band == -32767.] = np.nan
            value[:, :, i + len(bands)] = value_band

        f.close()

        # 将波段数据写入临时内存文件
        banddes = bands + navi
        image: gdal.Dataset = write_bands(value, banddes=banddes)

        # 控制点列表, 设置7*7个控制点
        gcps = []
        x_arr = np.linspace(0, Longitude.shape[1] - 1, num=7, endpoint=True, dtype=np.int)
        y_arr = np.linspace(0, Longitude.shape[0] - 1, num=7, endpoint=True, dtype=np.int)
        for x in x_arr:
            for y in y_arr:
                if abs(Longitude[y, x]) > 180 or abs(Latitude[y, x]) > 90:
                    continue
                gcps.append(gdal.GCP(np.float64(Longitude[y, x]), np.float64(Latitude[y, x]),
                                     0,
                                     np.float64(x), np.float64(y)))

        # 设置空间参考
        sr = osr.SpatialReference()
        sr.ImportFromEPSG(4326)
        sr.MorphToESRI()

        # 给数据及设置控制点及空间参考
        image.SetGCPs(gcps, sr.ExportToWkt())

        outfile = os.path.dirname(infile) + os.sep + os.path.basename(infile)[0:-3] + '.geotiff'

        cutlinelayer = bands + navi
        # 校正
        dst = gdal.Warp(outfile, image, format='GTiff', tps=True, xRes=0.01, yRes=0.01, dstNodata=np.nan,
                        resampleAlg=gdal.GRA_NearestNeighbour, cutlineLayer=cutlinelayer)  # dstNodata=65535
        for i, bandname in enumerate(cutlinelayer):
            band = dst.GetRasterBand(i + 1)
            band.SetMetadata({'bandname': bandname})
            band.SetDescription(bandname)

        image: None
        return outfile

    @staticmethod
    def H1B_simula(infile=None, bands=None):
        f = h5py.File(infile, "r")
        Longitude = f['/Navigation Data/lon'][:, :]
        Latitude = f['/Navigation Data/lat'][:, :]
        bands = ['Lt_simu_412', 'Lt_simu_443', 'Lt_simu_490', 'Lt_simu_520', 'Lt_simu_565', 'Lt_simu_670',
                 'Lt_simu_750', 'Lt_simu_865', 'DN_412', 'DN_443', 'DN_490', 'DN_520', 'DN_565', 'DN_670',
                 'DN_750', 'DN_865', 'sza_targetsensor', 'vza_targetsensor', 'saa_targetsensor', 'vaa_targetsensor']

        value = np.empty((Longitude.shape[0], Longitude.shape[1], len(bands)))
        for i, band in enumerate(bands):
            dataset_band = f['/Geophysical Data/' + bands[i]]
            value_band = dataset_band[:, :] * 1.
            value_band[value_band == -32767.] = np.nan
            value[:, :, i] = value_band

        f.close()

        # 将波段数据写入临时内存文件
        banddes = bands
        image: gdal.Dataset = write_bands(value, banddes=banddes)

        # 控制点列表, 设置7*7个控制点
        gcps = []
        x_arr = np.linspace(0, Longitude.shape[1] - 1, num=7, endpoint=True, dtype=np.int)
        y_arr = np.linspace(0, Longitude.shape[0] - 1, num=7, endpoint=True, dtype=np.int)
        for x in x_arr:
            for y in y_arr:
                if abs(Longitude[y, x]) > 180 or abs(Latitude[y, x]) > 90:
                    continue
                gcps.append(gdal.GCP(np.float64(Longitude[y, x]), np.float64(Latitude[y, x]),
                                     0,
                                     np.float64(x), np.float64(y)))

        # 设置空间参考
        sr = osr.SpatialReference()
        sr.ImportFromEPSG(4326)
        sr.MorphToESRI()

        # 给数据及设置控制点及空间参考
        image.SetGCPs(gcps, sr.ExportToWkt())

        outfile = os.path.dirname(infile) + os.sep + os.path.basename(infile)[0:-3] + '.geotiff'

        cutlinelayer = bands
        # 校正
        dst = gdal.Warp(outfile, image, format='GTiff', tps=True, xRes=0.01, yRes=0.01, dstNodata=np.nan,
                        resampleAlg=gdal.GRA_NearestNeighbour, cutlineLayer=cutlinelayer)  # dstNodata=65535
        for i, bandname in enumerate(cutlinelayer):
            band = dst.GetRasterBand(i + 1)
            band.SetMetadata({'bandname': bandname})
            band.SetDescription(bandname)

        image: None
        return outfile

    @staticmethod
    def H1A_simula(infile=None, bands=None):
        f = h5py.File(infile, "r")
        Longitude = f['/Navigation Data/lon'][:, :]
        Latitude = f['/Navigation Data/lat'][:, :]
        bands = ['Lt_simu_412', 'Lt_simu_443', 'Lt_simu_490', 'Lt_simu_520', 'Lt_simu_565', 'Lt_simu_670',
                 'Lt_simu_750', 'Lt_simu_865', 'DN_412', 'DN_443', 'DN_490', 'DN_520', 'DN_565', 'DN_670',
                 'DN_750', 'DN_865', 'sza_targetsensor', 'vza_targetsensor', 'saa_targetsensor', 'vaa_targetsensor']

        value = np.empty((Longitude.shape[0], Longitude.shape[1], len(bands)))
        for i, band in enumerate(bands):
            dataset_band = f['/Geophysical Data/' + bands[i]]
            value_band = dataset_band[:, :] * 1.
            value_band[value_band == -32767.] = np.nan
            value[:, :, i] = value_band

        f.close()

        # 将波段数据写入临时内存文件
        banddes = bands
        image: gdal.Dataset = write_bands(value, banddes=banddes)

        # 控制点列表, 设置7*7个控制点
        gcps = []
        x_arr = np.linspace(0, Longitude.shape[1] - 1, num=7, endpoint=True, dtype=np.int)
        y_arr = np.linspace(0, Longitude.shape[0] - 1, num=7, endpoint=True, dtype=np.int)
        for x in x_arr:
            for y in y_arr:
                if abs(Longitude[y, x]) > 180 or abs(Latitude[y, x]) > 90:
                    continue
                gcps.append(gdal.GCP(np.float64(Longitude[y, x]), np.float64(Latitude[y, x]),
                                     0,
                                     np.float64(x), np.float64(y)))

        # 设置空间参考
        sr = osr.SpatialReference()
        sr.ImportFromEPSG(4326)
        sr.MorphToESRI()

        # 给数据及设置控制点及空间参考
        image.SetGCPs(gcps, sr.ExportToWkt())

        outfile = os.path.dirname(infile) + os.sep + os.path.basename(infile)[0:-3] + '.geotiff'

        cutlinelayer = bands
        # 校正
        dst = gdal.Warp(outfile, image, format='GTiff', tps=True, xRes=0.01, yRes=0.01, dstNodata=np.nan,
                        resampleAlg=gdal.GRA_NearestNeighbour, cutlineLayer=cutlinelayer)  # dstNodata=65535
        for i, bandname in enumerate(cutlinelayer):
            band = dst.GetRasterBand(i + 1)
            band.SetMetadata({'bandname': bandname})
            band.SetDescription(bandname)

        image: None
        return outfile


class modis:
    @staticmethod
    def l2a(infile=None,
            radi_bands=['Rrs_412', 'Rrs_443', 'Rrs_469', 'Rrs_488', 'Rrs_531', 'Rrs_547', 'Rrs_555', 'Rrs_645',
                        'Rrs_667', 'Rrs_678', 'Rrs_748', 'Rrs_859', 'Rrs_869', 'chlor_a'], latlon=True):
        try:
            f = h5py.File(infile, "a")
            longitude = f['/navigation_data/longitude'][()]
            latitude = f['/navigation_data/latitude'][()]

            if latlon:
                value = np.empty((longitude.shape[0], longitude.shape[1], len(radi_bands) + 2))
            else:
                value = np.empty((longitude.shape[0], longitude.shape[1], len(radi_bands)))
            for i in range(radi_bands.__len__()):
                dataset_band = f['/geophysical_data/' + radi_bands[i]]
                value_band = dataset_band[:, :] * 1.
                value_band[
                    ~((dataset_band.attrs['valid_min'][0] < value_band) & (
                            value_band < dataset_band.attrs['valid_max'][0]))] = np.nan
                try:
                    gain = dataset_band.attrs['scale_factor'][0]
                    offset = dataset_band.attrs['add_offset'][0]
                except:
                    gain = 1
                    offset = 0

                value_band = value_band * gain + offset
                value[:, :, i] = value_band
        except:
            return ''

        try:
            # 如果需要加入经纬度数据
            value[:, :, len(radi_bands)] = latitude
            value[:, :, len(radi_bands) + 1] = longitude
        except:
            pass
        f.close()

        # 将波段数据写入临时内存文件
        image: gdal.Dataset = write_bands(value)

        # 控制点列表, 设置7*7个控制点
        gcps = []
        x_arr = np.linspace(0, longitude.shape[1] - 1, num=7, endpoint=True, dtype=np.int)
        y_arr = np.linspace(0, longitude.shape[0] - 1, num=7, endpoint=True, dtype=np.int)
        for x in x_arr:
            for y in y_arr:
                if abs(longitude[y, x]) > 180 or abs(latitude[y, x]) > 90:
                    continue
                gcps.append(gdal.GCP(np.float64(longitude[y, x]), np.float64(latitude[y, x]),
                                     0,
                                     np.float64(x), np.float64(y)))

        # 设置空间参考
        sr = osr.SpatialReference()
        sr.ImportFromEPSG(4326)
        sr.MorphToESRI()

        # 给数据及设置控制点及空间参考
        image.SetGCPs(gcps, sr.ExportToWkt())

        outfile = os.path.dirname(infile) + os.sep + os.path.basename(infile)[0:-4] + '.geotiff'

        # 校正
        if latlon:
            cutlinelayer = radi_bands + ['latitude', 'longitude']
        else:
            cutlinelayer = radi_bands
        dst = gdal.Warp(outfile, image, format='GTiff', tps=True, xRes=0.01, yRes=0.01, dstNodata=np.nan,
                        resampleAlg=gdal.GRA_NearestNeighbour, cutlineLayer=cutlinelayer)  # dstNodata=65535

        for i, bandname in enumerate(cutlinelayer):
            band = dst.GetRasterBand(i + 1)
            band.SetMetadata({'bandname': bandname})
            band.SetDescription(bandname)
        image: None
        return outfile

    @staticmethod
    def l2a_rc(infile=None,
               radi_bands=['Lt_412', 'Lt_443', 'Lt_469', 'Lt_488', 'Lt_531', 'Lt_547', 'Lt_555', 'Lt_645', 'Lt_667',
                           'Lt_678', 'Lt_748', 'Lt_859', 'Lt_869'], latlon=True):
        F0 = [172.632, 187.484, 205.878, 195.117, 185.699, 186.475, 183.869, 157.811, 151.694, 147.470, 127.873, 97.174,
              95.816]
        f = h5py.File(infile, "r")
        longitude = f['/navigation_data/longitude'][()]
        latitude = f['/navigation_data/latitude'][()]

        if latlon:
            value = np.empty((longitude.shape[0], longitude.shape[1], len(radi_bands) + 2))
        else:
            value = np.empty((longitude.shape[0], longitude.shape[1], len(radi_bands)))

        # 天顶角
        sza_d = f['/geophysical_data/solz']
        value_band = sza_d[:, :] * 1.
        value_band[
            ~((sza_d.attrs['valid_min'][0] < value_band) & (
                    value_band < sza_d.attrs['valid_max'][0]))] = np.nan
        try:
            gain = sza_d.attrs['scale_factor'][0]
            offset = sza_d.attrs['add_offset'][0]
        except:
            gain = 1
            offset = 0
        sza = value_band * gain + offset

        # Lt
        for i in range(radi_bands.__len__()):
            dataset_band = f['/geophysical_data/' + radi_bands[i]]
            value_band = dataset_band[:, :] / 10.
            value_band[
                ~((dataset_band.attrs['valid_min'][0] < value_band) & (
                        value_band < dataset_band.attrs['valid_max'][0]))] = np.nan
            try:
                gain = dataset_band.attrs['scale_factor'][0]
                offset = dataset_band.attrs['add_offset'][0]
            except:
                gain = 1
                offset = 0
            value_band = value_band * gain + offset
            value1 = value_band

            dataset_band = f['/geophysical_data/' + 'Lr' + radi_bands[i][2:]]
            value_band = dataset_band[:, :] / 10.
            value_band[
                ~((dataset_band.attrs['valid_min'][0] < value_band) & (
                        value_band < dataset_band.attrs['valid_max'][0]))] = np.nan
            try:
                gain = dataset_band.attrs['scale_factor'][0]
                offset = dataset_band.attrs['add_offset'][0]
            except:
                gain = 1
                offset = 0
            value_band = value_band * gain + offset
            value2 = value_band

            value[:, :, i] = (value1 - value2) * np.pi / (np.cos(sza * np.pi / 180)) / F0[i] / np.pi

        try:
            # 如果需要加入经纬度数据
            value[:, :, len(radi_bands)] = latitude
            value[:, :, len(radi_bands) + 1] = longitude
        except:
            pass
        f.close()

        # 将波段数据写入临时内存文件
        image: gdal.Dataset = write_bands(value)

        # 控制点列表, 设置7*7个控制点
        gcps = []
        x_arr = np.linspace(0, longitude.shape[1] - 1, num=7, endpoint=True, dtype=np.int)
        y_arr = np.linspace(0, longitude.shape[0] - 1, num=7, endpoint=True, dtype=np.int)
        for x in x_arr:
            for y in y_arr:
                if abs(longitude[y, x]) > 180 or abs(latitude[y, x]) > 90:
                    continue
                gcps.append(gdal.GCP(np.float64(longitude[y, x]), np.float64(latitude[y, x]),
                                     0,
                                     np.float64(x), np.float64(y)))

        # 设置空间参考
        sr = osr.SpatialReference()
        sr.ImportFromEPSG(4326)
        sr.MorphToESRI()

        # 给数据及设置控制点及空间参考
        image.SetGCPs(gcps, sr.ExportToWkt())

        outfile = os.path.dirname(infile) + os.sep + os.path.basename(infile)[0:-4] + '.geotiff'

        # 校正
        rc = ['Rrc_412', 'Rrc_443', 'Rrc_469', 'Rrc_488', 'Rrc_531', 'Rrc_547', 'Rrc_555', 'Rrc_645', 'Rrc_667',
              'Rrc_678', 'Rrc_748', 'Rrc_859', 'Rrc_869']
        if latlon:
            cutlinelayer = rc + ['latitude', 'longitude']
        else:
            cutlinelayer = rc
        dst = gdal.Warp(outfile, image, format='GTiff', tps=True, xRes=0.01, yRes=0.01, dstNodata=np.nan,
                        resampleAlg=gdal.GRA_NearestNeighbour, cutlineLayer=cutlinelayer)  # dstNodata=65535

        for i, bandname in enumerate(cutlinelayer):
            band = dst.GetRasterBand(i + 1)
            band.SetMetadata({'bandname': bandname})
            band.SetDescription(bandname)
        image: None
        return outfile


if __name__ == '__main__':
    print('用于各种数据转geotiff格式')
    # infile = r'D:\HYproject\fifthRound\highQualityimages\HY1B/H1B_OPER_OCT_L1A_20080305T024100_20080305T024100_04696_10.h5'
    # oct.l1agcp(infile=infile)

    # files = glob.glob(r'G:\high_quality\H1B_70images\crosscalibration/H1B_RICH_OCT_L1A*crossCalibration.h5')
    # for file in files:
    #     oct.H1B_simula(infile=file)

    files = glob.glob(r'G:\high_quality\H1A_26images\calibration_h1a\OCT\select/H1A_RICH_OCT_L1B*crossCalibration.h5')
    for file in files:
        oct.H1A_simula(infile=file)
