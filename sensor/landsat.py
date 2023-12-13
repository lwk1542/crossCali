# -*- coding: utf-8 -*-

# Corrects Landsat data, files must be original files with all bands in tar.gz format
#

import math
import os
import shutil
import tarfile

import numpy as np

import Functions as funcs


############################################################################
# All of the follwing are used to calculate earth-sun distance, adapted from
# ttp://davit.ece.vt.edu/davitpy/_modules/utils/calcSun.html#calcSunRadVector
def calcGeomMeanAnomalySun(t):
    """ Calculate the Geometric Mean Anomaly of the Sun (in degrees)     """
    M = 357.52911 + t * (35999.05029 - 0.0001537 * t)
    return M  # in degrees


def calcEccentricityEarthOrbit(t):
    """ Calculate the eccentricity of earth's orbit (unitless)   """
    e = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)
    return e  # unitless


def calcSunEqOfCenter(t):
    """Calculate the equation of center for the sun (in degrees) """
    mrad = np.radians(calcGeomMeanAnomalySun(t))
    sinm = np.sin(mrad)
    sin2m = np.sin(mrad + mrad)
    sin3m = np.sin(mrad + mrad + mrad)
    C = (sinm * (1.914602 - t * (0.004817 + 0.000014 * t)) +
         sin2m * (0.019993 - 0.000101 * t) + sin3m * 0.000289)
    return C  # in degrees


def calcSunTrueAnomaly(t):
    m = calcGeomMeanAnomalySun(t)
    c = calcSunEqOfCenter(t)
    v = m + c
    return v  # in degrees


def ES_dist(t):
    # eccent = 0.01672592        # Eccentricity of Earth orbit
    # axsmaj = 1.4957            # Semi-major axis of Earth orbit (km)
    # solyr  = 365.2563855       # Number of days in a solar year

    v = calcSunTrueAnomaly(t)
    e = calcEccentricityEarthOrbit(t)
    R = (1.000001018 * (1. - e * e)) / (1. + e * np.cos(np.radians(v)))
    return R


############################################################################


# Define functions for spectral indices
def NDVI(R, NIR):
    # LS8 = B4,B5    LS57 = B3,B4
    ndvi = (NIR - R) / (NIR + R)
    ndvi = ndvi.astype(np.float32)
    return ndvi


def EVI(B, R, NIR, L=1):
    # LS8 = B2,B4,B5    LS57 = B1,B3,B4
    evi = (NIR - R) / (NIR + 6 * R - 7.5 * B + L)
    evi = evi.astype(np.float32)
    return evi


def SAVI(R, NIR, L=0.5):
    # LS8 = B4,B5    LS57 = B3,B4
    savi = ((NIR - R) / (NIR + R + L)) * (1 + L)
    savi = savi.astype(np.float32)
    return savi


def MSAVI(R, NIR):
    # LS8 = B4,B5    LS57 = B3,B4
    msavi = (2 * NIR + 1 - np.sqrt((2 * NIR + 1) ** 2 - 8 * (NIR - R))) / 2
    msavi = msavi.astype(np.float32)
    return msavi


def NDMI(NIR, SWIR):
    # LS8 = B5,B6    LS57 = B4,B5
    ndmi = (NIR - SWIR) / (NIR + SWIR)
    ndmi = ndmi.astype(np.float32)
    return ndmi


def NBR(NIR, SWIR):
    # LS8 = B5,B7    LS57 = B4,B7
    nbr = (NIR - SWIR) / (NIR + SWIR)
    nbr = nbr.astype(np.float32)
    return nbr


def NBR2(SWIR1, SWIR2):
    # LS8 = B6,B7      LS57 = B5,B7
    nbr2 = (SWIR1 - SWIR2) / (SWIR1 + SWIR2)
    nbr2 = nbr2.astype(np.float32)
    return nbr2


def NDSI(G, SWIR):
    # LS8 = B3,B6      LS57 = B2,B5
    ndsi = (G - SWIR) / (G + SWIR)
    ndsi = ndsi.astype(np.float32)
    return ndsi


############################################################################

# returns a dictionary containing the metadata from the MTL file
def metaData(curFol, LSname):
    # Find the metafile and extract variables
    for x in os.listdir(curFol):

        if x[-7:].lower() == "mtl.txt":

            # Doesn't matter which Landsat, dictionary is only filled with exisiting keywords
            metaFile = open(curFol + x, "r")

            # Dictionary to store all metafile infos
            metaDict = {}

            # Name of current Landsat file, extracted from first band name
            metaDict['julDay'] = int(LSname[13:16])
            metaDict['year'] = int(LSname[17:21])

            for lines in metaFile.readlines():
                # lines still contain whitespaces
                # strLine = lines.decode("utf-8").replace(" ", "")
                strLine = lines.replace(" ", "")

                if strLine[:13] == 'SPACECRAFT_ID':
                    metaDict['spacecraft_ID'] = strLine[15:-2]
                if strLine[:8] == 'WRS_PATH':
                    metaDict['wrs_path'] = strLine[9:]
                if strLine[:7] == 'WRS_ROW':
                    metaDict['wrs_row'] = strLine[8:]
                if strLine[:11] == 'CLOUD_COVER':
                    metaDict['cloud_cover'] = float(strLine[12:-1])
                if strLine[:13] == 'DATE_ACQUIRED':
                    metaDict['date_aquired_str'] = strLine[14:-1]
                if strLine[:17] == 'SCENE_CENTER_TIME':
                    metaDict['scene_center_time_str'] = strLine[18:-1]
                if strLine[:8] == 'UTM_ZONE':
                    metaDict['utm_zone'] = strLine[9:-1]

                if strLine[:11] == 'SUN_AZIMUTH':
                    metaDict['sun_azimuth'] = float(strLine[12:])
                if strLine[:13] == 'SUN_ELEVATION':
                    metaDict['sun_elevation'] = float(strLine[14:])
                if strLine[:18] == 'EARTH_SUN_DISTANCE':
                    metaDict['earth_sun_distance'] = float(strLine[19:])

                # Radiances, first ones need to make sure Band_10 or Band_11 are not chosen instead
                if strLine[:23] == 'RADIANCE_MAXIMUM_BAND_1' and strLine[23] == "=":
                    metaDict['radiance_max_B1'] = float(strLine[24:-2])
                if strLine[:23] == 'RADIANCE_MINIMUM_BAND_1' and strLine[23] == "=":
                    metaDict['radiance_min_B1'] = float(strLine[24:-2])
                if strLine[:23] == 'RADIANCE_MAXIMUM_BAND_2':
                    metaDict['radiance_max_B2'] = float(strLine[24:-2])
                if strLine[:23] == 'RADIANCE_MINIMUM_BAND_2':
                    metaDict['radiance_min_B2'] = float(strLine[24:-2])
                if strLine[:23] == 'RADIANCE_MAXIMUM_BAND_3':
                    metaDict['radiance_max_B3'] = float(strLine[24:-2])
                if strLine[:23] == 'RADIANCE_MINIMUM_BAND_3':
                    metaDict['radiance_min_B3'] = float(strLine[24:-2])
                if strLine[:23] == 'RADIANCE_MAXIMUM_BAND_4':
                    metaDict['radiance_max_B4'] = float(strLine[24:-2])
                if strLine[:23] == 'RADIANCE_MINIMUM_BAND_4':
                    metaDict['radiance_min_B4'] = float(strLine[24:-2])
                if strLine[:23] == 'RADIANCE_MAXIMUM_BAND_5':
                    metaDict['radiance_max_B5'] = float(strLine[24:-2])
                if strLine[:23] == 'RADIANCE_MINIMUM_BAND_5':
                    metaDict['radiance_min_B5'] = float(strLine[24:-2])

                if LSname[2] == '8':
                    if strLine[:23] == 'RADIANCE_MAXIMUM_BAND_6':
                        metaDict['radiance_max_B6'] = float(strLine[24:-2])
                    if strLine[:23] == 'RADIANCE_MINIMUM_BAND_6':
                        metaDict['radiance_min_B6'] = float(strLine[24:-2])
                if LSname[2] == '7':
                    if strLine[:30] == 'RADIANCE_MAXIMUM_BAND_6_VCID_1':
                        metaDict['radiance_max_B61'] = float(strLine[31:-2])
                    if strLine[:30] == 'RADIANCE_MINIMUM_BAND_6_VCID_1':
                        metaDict['radiance_min_B61'] = float(strLine[31:-2])
                    if strLine[:30] == 'RADIANCE_MAXIMUM_BAND_6_VCID_2':
                        metaDict['radiance_max_B62'] = float(strLine[31:-2])
                    if strLine[:30] == 'RADIANCE_MINIMUM_BAND_6_VCID_2':
                        metaDict['radiance_min_B62'] = float(strLine[31:-2])
                if LSname[2] == '5':
                    if strLine[:23] == 'RADIANCE_MAXIMUM_BAND_6':
                        metaDict['radiance_max_B6'] = float(strLine[24:-2])
                    if strLine[:23] == 'RADIANCE_MINIMUM_BAND_6':
                        metaDict['radiance_min_B6'] = float(strLine[24:-2])

                if strLine[:23] == 'RADIANCE_MAXIMUM_BAND_7':
                    metaDict['radiance_max_B7'] = float(strLine[24:-2])
                if strLine[:23] == 'RADIANCE_MINIMUM_BAND_7':
                    metaDict['radiance_min_B7'] = float(strLine[24:-2])
                if strLine[:23] == 'RADIANCE_MAXIMUM_BAND_8':
                    metaDict['radiance_max_B8'] = float(strLine[24:-2])
                if strLine[:23] == 'RADIANCE_MINIMUM_BAND_8':
                    metaDict['radiance_min_B8'] = float(strLine[24:-2])
                if strLine[:23] == 'RADIANCE_MAXIMUM_BAND_9':
                    metaDict['radiance_max_B9'] = float(strLine[24:-2])
                if strLine[:23] == 'RADIANCE_MINIMUM_BAND_9':
                    metaDict['radiance_min_B9'] = float(strLine[24:-2])

                if strLine[:24] == 'RADIANCE_MAXIMUM_BAND_10':
                    metaDict['radiance_max_B10'] = float(strLine[25:-2])
                if strLine[:24] == 'RADIANCE_MINIMUM_BAND_10':
                    metaDict['radiance_min_B10'] = float(strLine[25:-2])
                if strLine[:24] == 'RADIANCE_MAXIMUM_BAND_11':
                    metaDict['radiance_max_B11'] = float(strLine[25:-2])
                if strLine[:24] == 'RADIANCE_MINIMUM_BAND_11':
                    metaDict['radiance_min_B11'] = float(strLine[25:-2])

                # Reflectances, only 9 bands in metafile
                if strLine[:26] == 'REFLECTANCE_MAXIMUM_BAND_1':
                    metaDict['reflectance_max_B1'] = float(strLine[27:-2])
                if strLine[:26] == 'REFLECTANCE_MINIMUM_BAND_1':
                    metaDict['reflectance_min_B1'] = float(strLine[27:-2])
                if strLine[:26] == 'REFLECTANCE_MAXIMUM_BAND_2':
                    metaDict['reflectance_max_B2'] = float(strLine[27:-2])
                if strLine[:26] == 'REFLECTANCE_MINIMUM_BAND_2':
                    metaDict['reflectance_min_B2'] = float(strLine[27:-2])
                if strLine[:26] == 'REFLECTANCE_MAXIMUM_BAND_3':
                    metaDict['reflectance_max_B3'] = float(strLine[27:-2])
                if strLine[:26] == 'REFLECTANCE_MINIMUM_BAND_3':
                    metaDict['reflectance_min_B3'] = float(strLine[27:-2])
                if strLine[:26] == 'REFLECTANCE_MAXIMUM_BAND_4':
                    metaDict['reflectance_max_B4'] = float(strLine[27:-2])
                if strLine[:26] == 'REFLECTANCE_MINIMUM_BAND_4':
                    metaDict['reflectance_min_B4'] = float(strLine[27:-2])
                if strLine[:26] == 'REFLECTANCE_MAXIMUM_BAND_5':
                    metaDict['reflectance_max_B5'] = float(strLine[27:-2])
                if strLine[:26] == 'REFLECTANCE_MINIMUM_BAND_5':
                    metaDict['reflectance_min_B5'] = float(strLine[27:-2])
                if strLine[:26] == 'REFLECTANCE_MAXIMUM_BAND_6':
                    metaDict['reflectance_max_B6'] = float(strLine[27:-2])
                if strLine[:26] == 'REFLECTANCE_MINIMUM_BAND_6':
                    metaDict['reflectance_min_B6'] = float(strLine[27:-2])
                if strLine[:26] == 'REFLECTANCE_MAXIMUM_BAND_7':
                    metaDict['reflectance_max_B7'] = float(strLine[27:-2])
                if strLine[:26] == 'REFLECTANCE_MINIMUM_BAND_7':
                    metaDict['reflectance_min_B7'] = float(strLine[27:-2])
                if strLine[:26] == 'REFLECTANCE_MAXIMUM_BAND_8':
                    metaDict['reflectance_max_B8'] = float(strLine[27:-2])
                if strLine[:26] == 'REFLECTANCE_MINIMUM_BAND_8':
                    metaDict['reflectance_min_B8'] = float(strLine[27:-2])
                if strLine[:26] == 'REFLECTANCE_MAXIMUM_BAND_9':
                    metaDict['reflectance_max_B9'] = float(strLine[27:-2])
                if strLine[:26] == 'REFLECTANCE_MINIMUM_BAND_9':
                    metaDict['reflectance_min_B9'] = float(strLine[27:-2])

                # Min_Max pixel value
                if strLine[:23] == 'QUANTIZE_CAL_MAX_BAND_1' and strLine[23] == "=":
                    metaDict['quant_max_B1'] = int(strLine[24:])
                if strLine[:23] == 'QUANTIZE_CAL_MIN_BAND_1' and strLine[23] == "=":
                    metaDict['quant_min_B1'] = int(strLine[24:])
                if strLine[:23] == 'QUANTIZE_CAL_MAX_BAND_2':
                    metaDict['quant_max_B2'] = int(strLine[24:])
                if strLine[:23] == 'QUANTIZE_CAL_MIN_BAND_2':
                    metaDict['quant_min_B2'] = int(strLine[24:])
                if strLine[:23] == 'QUANTIZE_CAL_MAX_BAND_3':
                    metaDict['quant_max_B3'] = int(strLine[24:])
                if strLine[:23] == 'QUANTIZE_CAL_MIN_BAND_3':
                    metaDict['quant_min_B3'] = int(strLine[24:])
                if strLine[:23] == 'QUANTIZE_CAL_MAX_BAND_4':
                    metaDict['quant_max_B4'] = int(strLine[24:])
                if strLine[:23] == 'QUANTIZE_CAL_MIN_BAND_4':
                    metaDict['quant_min_B4'] = int(strLine[24:])
                if strLine[:23] == 'QUANTIZE_CAL_MAX_BAND_5':
                    metaDict['quant_max_B5'] = int(strLine[24:])
                if strLine[:23] == 'QUANTIZE_CAL_MIN_BAND_5':
                    metaDict['quant_min_B5'] = int(strLine[24:])

                if LSname[2] == '8' or LSname[2] == '5':
                    if strLine[:23] == 'QUANTIZE_CAL_MAX_BAND_6':
                        metaDict['quant_max_B6'] = int(strLine[24:])
                    if strLine[:23] == 'QUANTIZE_CAL_MIN_BAND_6':
                        metaDict['quant_min_B6'] = int(strLine[24:])
                if LSname[2] == '7':
                    if strLine[:30] == 'QUANTIZE_CAL_MAX_BAND_6_VCID_1':
                        metaDict['quant_max_B61'] = int(strLine[31:])
                    if strLine[:30] == 'QUANTIZE_CAL_MIN_BAND_6_VCID_1':
                        metaDict['quant_min_B61'] = int(strLine[31:])
                    if strLine[:30] == 'QUANTIZE_CAL_MAX_BAND_6_VCID_2':
                        metaDict['quant_max_B62'] = int(strLine[31:])
                    if strLine[:30] == 'QUANTIZE_CAL_MIN_BAND_6_VCID_2':
                        metaDict['quant_min_B62'] = int(strLine[31:])

                if strLine[:23] == 'QUANTIZE_CAL_MAX_BAND_7':
                    metaDict['quant_max_B7'] = int(strLine[24:])
                if strLine[:23] == 'QUANTIZE_CAL_MIN_BAND_7':
                    metaDict['quant_min_B7'] = int(strLine[24:])
                if strLine[:23] == 'QUANTIZE_CAL_MAX_BAND_8':
                    metaDict['quant_max_B8'] = int(strLine[24:])
                if strLine[:23] == 'QUANTIZE_CAL_MIN_BAND_8':
                    metaDict['quant_min_B8'] = int(strLine[24:])
                if strLine[:23] == 'QUANTIZE_CAL_MAX_BAND_9':
                    metaDict['quant_max_B9'] = int(strLine[24:])
                if strLine[:23] == 'QUANTIZE_CAL_MIN_BAND_9':
                    metaDict['quant_min_B9'] = int(strLine[24:])

                if strLine[:24] == 'QUANTIZE_CAL_MAX_BAND_10':
                    metaDict['quant_max_B10'] = int(strLine[25:])
                if strLine[:24] == 'QUANTIZE_CAL_MIN_BAND_10':
                    metaDict['quant_min_B10'] = int(strLine[25:])
                if strLine[:24] == 'QUANTIZE_CAL_MAX_BAND_11':
                    metaDict['quant_max_B11'] = int(strLine[25:])
                if strLine[:24] == 'QUANTIZE_CAL_MIN_BAND_11':
                    metaDict['quant_min_B11'] = int(strLine[25:])

                # Radiometric Rescaling
                if strLine[:20] == 'RADIANCE_MULT_BAND_1' and strLine[20] == "=":
                    metaDict['radiance_mult_B1'] = float(strLine[21:])
                if strLine[:20] == 'RADIANCE_MULT_BAND_2':
                    metaDict['radiance_mult_B2'] = float(strLine[21:])
                if strLine[:20] == 'RADIANCE_MULT_BAND_3':
                    metaDict['radiance_mult_B3'] = float(strLine[21:])
                if strLine[:20] == 'RADIANCE_MULT_BAND_4':
                    metaDict['radiance_mult_B4'] = float(strLine[21:])
                if strLine[:20] == 'RADIANCE_MULT_BAND_5':
                    metaDict['radiance_mult_B5'] = float(strLine[21:])

                if LSname[2] == '8' or LSname[2] == '5':
                    if strLine[:20] == 'RADIANCE_MULT_BAND_6':
                        metaDict['radiance_mult_B6'] = float(strLine[21:])
                if LSname[2] == '7':
                    if strLine[:27] == 'RADIANCE_MULT_BAND_6_VCID_1':
                        metaDict['radiance_mult_B61'] = float(strLine[28:])
                    if strLine[:27] == 'RADIANCE_MULT_BAND_6_VCID_2':
                        metaDict['radiance_mult_B62'] = float(strLine[28:])

                if strLine[:20] == 'RADIANCE_MULT_BAND_7':
                    metaDict['radiance_mult_B7'] = float(strLine[21:])
                if strLine[:20] == 'RADIANCE_MULT_BAND_8':
                    metaDict['radiance_mult_B8'] = float(strLine[21:])
                if strLine[:20] == 'RADIANCE_MULT_BAND_9':
                    metaDict['radiance_mult_B9'] = float(strLine[21:])
                if strLine[:21] == 'RADIANCE_MULT_BAND_10':
                    metaDict['radiance_mult_B10'] = float(strLine[22:])
                if strLine[:21] == 'RADIANCE_MULT_BAND_11':
                    metaDict['radiance_mult_B11'] = float(strLine[22:])

                if strLine[:19] == 'RADIANCE_ADD_BAND_1' and strLine[19] == "=":
                    metaDict['radiance_add_B1'] = float(strLine[20:])
                if strLine[:19] == 'RADIANCE_ADD_BAND_2':
                    metaDict['radiance_add_B2'] = float(strLine[20:])
                if strLine[:19] == 'RADIANCE_ADD_BAND_3':
                    metaDict['radiance_add_B3'] = float(strLine[20:])
                if strLine[:19] == 'RADIANCE_ADD_BAND_4':
                    metaDict['radiance_add_B4'] = float(strLine[20:])
                if strLine[:19] == 'RADIANCE_ADD_BAND_5':
                    metaDict['radiance_add_B5'] = float(strLine[20:])

                if LSname[2] == '8' or LSname[2] == '5':
                    if strLine[:19] == 'RADIANCE_ADD_BAND_6':
                        metaDict['radiance_add_B6'] = float(strLine[20:])
                if LSname[2] == '7':
                    if strLine[:26] == 'RADIANCE_ADD_BAND_6_VCID_1':
                        metaDict['radiance_add_B61'] = float(strLine[27:])
                    if strLine[:26] == 'RADIANCE_ADD_BAND_6_VCID_2':
                        metaDict['radiance_add_B62'] = float(strLine[27:])

                if strLine[:19] == 'RADIANCE_ADD_BAND_7':
                    metaDict['radiance_add_B7'] = float(strLine[20:])
                if strLine[:19] == 'RADIANCE_ADD_BAND_8':
                    metaDict['radiance_add_B8'] = float(strLine[20:])
                if strLine[:19] == 'RADIANCE_ADD_BAND_9':
                    metaDict['radiance_add_B9'] = float(strLine[20:])
                if strLine[:20] == 'RADIANCE_ADD_BAND_10':
                    metaDict['radiance_add_B10'] = float(strLine[21:])
                if strLine[:20] == 'RADIANCE_ADD_BAND_11':
                    metaDict['radiance_add_B11'] = float(strLine[21:])

                if strLine[:23] == 'REFLECTANCE_MULT_BAND_1' and strLine[23] == "=":
                    metaDict['reflectance_mult_B1'] = float(strLine[24:])
                if strLine[:23] == 'REFLECTANCE_MULT_BAND_2':
                    metaDict['reflectance_mult_B2'] = float(strLine[24:])
                if strLine[:23] == 'REFLECTANCE_MULT_BAND_3':
                    metaDict['reflectance_mult_B3'] = float(strLine[24:])
                if strLine[:23] == 'REFLECTANCE_MULT_BAND_4':
                    metaDict['reflectance_mult_B4'] = float(strLine[24:])
                if strLine[:23] == 'REFLECTANCE_MULT_BAND_5':
                    metaDict['reflectance_mult_B5'] = float(strLine[24:])
                if strLine[:23] == 'REFLECTANCE_MULT_BAND_6':
                    metaDict['reflectance_mult_B6'] = float(strLine[24:])
                if strLine[:23] == 'REFLECTANCE_MULT_BAND_7':
                    metaDict['reflectance_mult_B7'] = float(strLine[24:])
                if strLine[:23] == 'REFLECTANCE_MULT_BAND_8':
                    metaDict['reflectance_mult_B8'] = float(strLine[24:])
                if strLine[:23] == 'REFLECTANCE_MULT_BAND_9':
                    metaDict['reflectance_mult_B9'] = float(strLine[24:])

                if strLine[:22] == 'REFLECTANCE_ADD_BAND_1' and strLine[22] == "=":
                    metaDict['reflectance_add_B1'] = float(strLine[23:])
                if strLine[:22] == 'REFLECTANCE_ADD_BAND_2':
                    metaDict['reflectance_add_B2'] = float(strLine[23:])
                if strLine[:22] == 'REFLECTANCE_ADD_BAND_3':
                    metaDict['reflectance_add_B3'] = float(strLine[23:])
                if strLine[:22] == 'REFLECTANCE_ADD_BAND_4':
                    metaDict['reflectance_add_B4'] = float(strLine[23:])
                if strLine[:22] == 'REFLECTANCE_ADD_BAND_5':
                    metaDict['reflectance_add_B5'] = float(strLine[23:])
                if strLine[:22] == 'REFLECTANCE_ADD_BAND_6':
                    metaDict['reflectance_add_B6'] = float(strLine[23:])
                if strLine[:22] == 'REFLECTANCE_ADD_BAND_7':
                    metaDict['reflectance_add_B7'] = float(strLine[23:])
                if strLine[:22] == 'REFLECTANCE_ADD_BAND_8':
                    metaDict['reflectance_add_B8'] = float(strLine[23:])
                if strLine[:22] == 'REFLECTANCE_ADD_BAND_9':
                    metaDict['reflectance_add_B9'] = float(strLine[23:])

                # Thermal Constants
                if strLine[:19] == 'K1_CONSTANT_BAND_10':
                    metaDict['k1_const_B10'] = float(strLine[20:])
                if strLine[:19] == 'K1_CONSTANT_BAND_11':
                    metaDict['k1_const_B11'] = float(strLine[20:])
                if strLine[:19] == 'K2_CONSTANT_BAND_10':
                    metaDict['k2_const_B10'] = float(strLine[20:])
                if strLine[:19] == 'K2_CONSTANT_BAND_11':
                    metaDict['k2_const_B11'] = float(strLine[20:])

                metaFile.close()

    return metaDict


# Calculate radiation and toa_reflectance
def rad_toa(LSname, curFol, metaDict):
    # creates a dict that will hold each bands/rasters name and corresponding arrays
    # done beforw with eval(varName = create.array), this strangly didn't work for
    # all band in Python 3.5
    rrdict = {}

    # Handle Landsat 8 differently than 4,5 or 7
    if LSname[2] == '8':

        for x in range(1, 10):
            xStr = str(x)

            # read bands as arrays
            rrdict['arrayDN' + xStr] = funcs.singleTifToArray(curFol + LSname + "_B" + xStr + ".TIF")

            # convert to radiance and convert to 32-bit floating point for memory saving
            rrdict['lambda' + xStr] = metaDict['radiance_mult_B' + xStr] * rrdict['arrayDN' + xStr] + metaDict[
                'radiance_add_B' + xStr]
            rrdict['lambda' + xStr] = rrdict['lambda' + xStr].astype(np.float32)

            # convert to reflectance and convert to 32-bit floating point for memory saving
            rrdict['reflectance' + xStr] = ((metaDict['reflectance_mult_B' + xStr] * rrdict['arrayDN' + xStr] + \
                                             metaDict['reflectance_add_B' + xStr]) / \
                                            math.sin(math.radians(metaDict['sun_elevation'])))

            rrdict['reflectance' + xStr] = rrdict['reflectance' + xStr].astype(np.float32)

            del rrdict['arrayDN' + xStr]



    elif LSname[2] == '7' or LSname[2] == '5' or LSname[2] == '4':
        # calculate radiation and toa relfectance
        if LSname[2] == '7':
            lastBand = 9
        else:
            lastBand = 8

        for x in range(1, lastBand):
            xStr = str(x)
            # read bands as arrays, bands 6 need to be handled separately
            if x != 6:
                rrdict['arrayDN' + xStr] = funcs.singleTifToArray(curFol + LSname + "_B" + xStr + ".TIF")

                # convert to radiance and convert to 32-bit floating point for memory saving
                rrdict['lambda' + xStr] = ((metaDict['radiance_max_B' + xStr] - metaDict['radiance_min_B' + xStr]) / \
                                           (metaDict['quant_max_B' + xStr] - metaDict['quant_min_B' + xStr])) * \
                                          (rrdict['arrayDN' + xStr] - metaDict['quant_min_B' + xStr]) + \
                                          metaDict['radiance_min_B' + xStr]

                rrdict['lambda' + xStr] = rrdict['lambda' + xStr].astype(np.float32)

                # convert to reflectance and convert to 32-bit floating point for memory saving
                esun = [1970, 1842, 1547, 1044, 225.7, 0, 82.06, 1369][x - 1]  # band depending constant
                e_s_dist = ES_dist(metaDict["julDay"])

                rrdict['reflectance' + xStr] = (np.pi * rrdict['lambda' + xStr] * e_s_dist ** 2) / \
                                               (esun * \
                                                math.sin(math.radians(metaDict['sun_elevation'])))

                rrdict['reflectance' + xStr] = rrdict['reflectance' + xStr].astype(np.float32)

                del rrdict['arrayDN' + xStr]

    return rrdict


# Calculates surface temperature in Kelvin, returns a dictionary with arrays
def cKelvin(calcKelvin, LSname, curFol, metaDict):
    kDict = {}

    # Handle Landsat 8 differently than 5 or 7
    if LSname[2] == '8':
        # calculate temperature in Kelvin  from bands 10 and 11

        if calcKelvin == 1 or calcKelvin == 2:
            kDict['arrayDN10'] = funcs.singleTifToArray(curFol + LSname + "_B10.TIF")
            kDict['lambda10'] = metaDict['radiance_mult_B10'] * kDict['arrayDN10'] + metaDict['radiance_add_B10']
            kDict['lambda10'] = kDict['lambda10'].astype(np.float32)
            kDict['t10'] = metaDict['k2_const_B10'] / (
                np.log((metaDict['k1_const_B10'] / kDict['lambda10']) + 1))  # T in Kelvin
            kDict['t10'] = kDict['t10'].astype(np.float32)

            funcs.array_to_raster(curFol + LSname + "_B10.TIF", kDict['t10'],
                                  curFol + "Temp_B10.TIF")
            del kDict['t10']

        if calcKelvin == 1 or calcKelvin == 3:
            kDict['arrayDN11'] = funcs.singleTifToArray(curFol + LSname + "_B11.TIF")
            kDict['lambda11'] = metaDict['radiance_mult_B11'] * kDict['arrayDN11'] + metaDict['radiance_add_B11']
            kDict['lambda11'] = kDict['lambda11'].astype(np.float32)
            kDict['t11'] = metaDict['k2_const_B11'] / (np.log((metaDict['k1_const_B11'] / kDict['lambda11']) + 1))
            kDict['t11'] = kDict['t11'].astype(np.float32)

            funcs.array_to_raster(curFol + LSname + "_B11.TIF", kDict['t11'],
                                  curFol + "Temp_B11.TIF")
            del kDict['t11']


    elif LSname[2] == '7' or LSname[2] == '5' or LSname[2] == '4':

        if LSname[2] == '7':
            if calcKelvin == 1:
                startNum = 1
                endNum = 3
            elif calcKelvin == 2:
                startNum = 1
                endNum = 2
            else:
                startNum = 2
                endNum = 3

            for y in range(startNum, endNum):
                yStr = str(y)

                kDict['arrayDN6' + yStr] = funcs.singleTifToArray(curFol + LSname + "_B6_VCID_" + yStr + ".TIF")

                kDict['lambda6' + yStr] = (( \
                                                       metaDict['radiance_max_B6' + yStr] - metaDict[
                                                   'radiance_min_B6' + yStr]) / \
                                           (metaDict['quant_max_B6' + yStr] - metaDict['quant_min_B6' + yStr])) * \
                                          (kDict['arrayDN6' + yStr] - metaDict['quant_min_B6' + yStr]) + \
                                          metaDict['radiance_min_B6' + yStr]

                kDict['lambda6' + yStr] = kDict['lambda6' + yStr].astype(np.float32)

                k1_const_B61 = 666.09  # 607.76 for LS5
                k2_const_B62 = 1282.71  # 1260.56 for LS5

                kDict['t6' + yStr] = k2_const_B62 / \
                                     (np.log((k1_const_B61 / kDict['lambda6' + yStr]) + 1))
                kDict['t6' + yStr] = kDict['t6' + yStr].astype(np.float32)

                # del kDict['arrayDN6'+yStr]

                funcs.array_to_raster(curFol + LSname + \
                                      "_B6_VCID_" + yStr + ".TIF", kDict['t6' + yStr], curFol + \
                                      "Temp_B6" + yStr + ".TIF")

                del kDict['t6' + yStr]

        if LSname[2] == '5' or LSname[2] == '4':
            kDict['arrayDN6'] = funcs.singleTifToArray(
                curFol + LSname + "_B6.TIF")

            kDict['lambda6'] = (((metaDict['radiance_max_B6'] - metaDict['radiance_min_B6']) /
                                 (metaDict['quant_max_B6'] - metaDict['quant_min_B6'])) *
                                (kDict['arrayDN6'] - metaDict['quant_min_B6']) +
                                metaDict['radiance_min_B6'])
            kDict['lambda6'] = kDict['lambda6'].astype(np.float32)

            k1_const_B6 = 607.76
            k2_const_B6 = 1260.56

            kDict['t6'] = k2_const_B6 / np.log((k1_const_B6 / kDict['lambda6']) + 1)
            kDict['t6'] = kDict['t6'].astype(np.float32)

            # del kDict['arrayDN6']

            funcs.array_to_raster(curFol + LSname +
                                  "_B6.TIF", kDict['t6'], curFol + "Temp_B6.TIF")

            del kDict['t6']

    return kDict


# save radiation rasters to disk
def saveRadFunc(LSname, curFol, rrdict, calcKelvin):
    if LSname[2] == '8':

        if calcKelvin == 0:
            endRange = 10
        else:
            endRange = 12

        for x in range(1, endRange):
            xStr = str(x)
            funcs.array_to_raster(curFol + LSname + "_B" + xStr + ".TIF", rrdict['lambda' + xStr], \
                                  curFol + "Radiation_B" + xStr + ".TIF")
            # del rrdict['lambda'+xStr]


    elif LSname[2] == '7' or LSname[2] == '5' or LSname[2] == '4':

        if LSname[2] == '7':
            lastBand = 9
        else:
            lastBand = 8

        for x in range(1, lastBand):
            xStr = str(x)
            if LSname[2] == '7':
                if x != 6:
                    funcs.array_to_raster(curFol + LSname + "_B" + xStr + ".TIF", rrdict['lambda' + xStr], \
                                          curFol + "Radiation_B" + xStr + ".TIF")
                    # del rrdict['lambda'+xStr]

                else:
                    if calcKelvin != 0:
                        for y in range(1, 3):
                            yStr = str(y)
                            try:
                                funcs.array_to_raster(curFol + LSname + \
                                                      "_B" + xStr + "_VCID_" + yStr + ".TIF",
                                                      rrdict['lambda' + xStr + yStr], curFol + \
                                                      "Radiation_B" + xStr + yStr + ".TIF")
                                # del  rrdict['lambda'+xStr+yStr]

                            except:
                                pass

            if LSname[2] == '5' or LSname[2] == '4':
                if calcKelvin != 0:
                    funcs.array_to_raster(curFol + LSname + "_B" + xStr + ".TIF", rrdict['lambda' + xStr], \
                                          curFol + "Radiation_B" + xStr + ".TIF")
                # del rrdict['lambda'+xStr]


# Save reflectance rasters to disk
def saveReflFunc(LSname, curFol, rrdict):
    if LSname[2] == '8':
        for x in range(1, 10):
            xStr = str(x)
            funcs.array_to_raster(curFol + LSname + "_B" + xStr + ".TIF", rrdict['reflectance' + xStr], \
                                  curFol + "Reflectance_B" + xStr + ".TIF")

            # del rrdict['reflectance'+xStr]


    elif LSname[2] == '7' or LSname[2] == '5' or LSname[2] == '4':

        if LSname[2] == '7':
            lastBand = 9
        else:
            lastBand = 8

        for x in range(1, lastBand):
            xStr = str(x)
            if x != 6:
                funcs.array_to_raster(curFol + LSname + "_B" + xStr + ".TIF", \
                                      rrdict['reflectance' + xStr], curFol + "Reflectance_B" + xStr + ".TIF")
                # del rrdict['reflectance'+xStr]


# calculate spectral indices, returns a dictionary and also saves all tiffs to disk
def specIndFunc(LSname, curFol, allIndices, rrdict, saveInd=1):
    specDict = {}

    # NDVI
    if allIndices == 1 or allIndices == 2:
        if LSname[2] == '8':
            ndviAr = NDVI(rrdict['reflectance4'], rrdict['reflectance5'])
        elif LSname[2] == '7' or LSname[2] == '5' or LSname[2] == '4':
            ndviAr = NDVI(rrdict['reflectance3'], rrdict['reflectance4'])
        if saveInd == 1:
            funcs.array_to_raster(curFol + LSname + "_B4.TIF", ndviAr,
                                  curFol + "Ind_NDVI_" + LSname[:-5] + ".TIF")
        specDict['ndviAr'] = ndviAr

    if allIndices == 1 or allIndices == 3:
        if LSname[2] == '8':
            eviAr = EVI(rrdict['reflectance2'], rrdict['reflectance4'], rrdict['reflectance5'], L=1)
        elif LSname[2] == '7' or LSname[2] == '5' or LSname[2] == '4':
            eviAr = EVI(rrdict['reflectance1'], rrdict['reflectance3'], rrdict['reflectance4'], L=1)
        if saveInd == 1:
            funcs.array_to_raster(curFol + LSname + "_B2.TIF", eviAr,
                                  curFol + "Ind_EVI_" + LSname[:-5] + ".TIF")
        specDict['eviAr'] = eviAr

    if allIndices == 1 or allIndices == 4:
        if LSname[2] == '8':
            saviAr = SAVI(rrdict['reflectance4'], rrdict['reflectance5'], L=0.5)
        elif LSname[2] == '7' or LSname[2] == '5' or LSname[2] == '4':
            saviAr = SAVI(rrdict['reflectance3'], rrdict['reflectance4'], L=0.5)
        if saveInd == 1:
            funcs.array_to_raster(curFol + LSname + "_B4.TIF", saviAr,
                                  curFol + "Ind_SAVI_" + LSname[:-5] + ".TIF")
        specDict['saviAr'] = saviAr

    if allIndices == 1 or allIndices == 5:
        if LSname[2] == '8':
            msaviAr = MSAVI(rrdict['reflectance4'], rrdict['reflectance5'])
        elif LSname[2] == '7' or LSname[2] == '5' or LSname[2] == '4':
            msaviAr = MSAVI(rrdict['reflectance3'], rrdict['reflectance4'])
        if saveInd == 1:
            funcs.array_to_raster(curFol + LSname + "_B4.TIF", msaviAr,
                                  curFol + "Ind_MSAVI_" + LSname[:-5] + ".TIF")
        specDict['msaviAr'] = msaviAr

    if allIndices == 1 or allIndices == 6:
        if LSname[2] == '8':
            ndmiAr = NDMI(rrdict['reflectance5'], rrdict['reflectance6'])
        elif LSname[2] == '7' or LSname[2] == '5' or LSname[2] == '4':
            ndmiAr = NDMI(rrdict['reflectance4'], rrdict['reflectance5'])
        if saveInd == 1:
            funcs.array_to_raster(curFol + LSname + "_B5.TIF", ndmiAr,
                                  curFol + "Ind_NDMI_" + LSname[:-5] + ".TIF")
        specDict['ndmiAr'] = ndmiAr

    if allIndices == 1 or allIndices == 7:
        if LSname[2] == '8':
            nbrAr = NBR(rrdict['reflectance5'], rrdict['reflectance7'])
        elif LSname[2] == '7' or LSname[2] == '5' or LSname[2] == '4':
            nbrAr = NBR(rrdict['reflectance4'], rrdict['reflectance7'])
        if saveInd == 1:
            funcs.array_to_raster(curFol + LSname + "_B5.TIF", nbrAr,
                                  curFol + "Ind_NBR_" + LSname[:-5] + ".TIF")
        specDict['nbrAr'] = nbrAr

    if allIndices == 1 or allIndices == 8:
        if LSname[2] == '8':
            nbr2Ar = NBR2(rrdict['reflectance6'], rrdict['reflectance7'])
        elif LSname[2] == '7' or LSname[2] == '5' or LSname[2] == '4':
            nbr2Ar = NBR2(rrdict['reflectance5'], rrdict['reflectance7'])
        if saveInd == 1:
            funcs.array_to_raster(curFol + LSname + "_B7.TIF", nbr2Ar,
                                  curFol + "Ind_NBR2_" + LSname[:-5] + ".TIF")
        specDict['nbr2Ar'] = nbr2Ar

    if allIndices == 1 or allIndices == 9:
        if LSname[2] == '8':
            ndsi = NDSI(rrdict['reflectance3'], rrdict['reflectance6'])
        elif LSname[2] == '7' or LSname[2] == '5' or LSname[2] == '4':
            ndsi = NDSI(rrdict['reflectance2'], rrdict['reflectance5'])
        if saveInd == 1:
            funcs.array_to_raster(curFol + LSname + "_B7.TIF", ndsi,
                                  curFol + "Ind_NDSI_" + LSname[:-5] + ".TIF")
        specDict['ndsi'] = nbr2Ar

    return specDict


def Landsat(inFile, outFol, deleteOriginal, saveRadiation, saveReflectance, calcKelvin, allIndices):
    """
    Handles the calls for all above functions
    Basically fills the 'metaDict'dictionary with data from the Landsat MTL file and
    the 'rrdict' dictionary with arrays that may be saved to disk later

    :inFile:            a tar.gz file path

    :outFol:            output folder, here a subfolder with the LS name will be created
                        This folder will store all desired final tiffs

    :deleteOriginal:     Flag for deleting exracted inital DN Rasters
                         1 = keep all, 2 = delete all, 3 = delete but keep metafile
                         Note: folder needs to be empty at start, else all is deleted

    :saveRadiation:      if True Radiation Tiffs will be saved to disk

    :saveReflectance:    if True Reflectance Tiffs will be saved to disk

    :calcKelvin:         Flag for calculating temperature rasters in Kelvin
                         0 = no calculation, 1 = calc Band 10&11 or 61/62,
                         2 = calc B10/61 only, 3=calc B11/62 only

    :allIndices:         Flag for calculating all/one indices
                         # 0 = none, 1= all, 2 = NDVI, 3 = EVI, 4 = SAVI,
                         # 5 = MSAVI, 6 = NDMI, 7 = NBR, 8 = NBR2, 9 = NDSI
    :saveInd:     	  Flag if Indices should be saved to disk? 0 = no, 1 = yes (default)


    """

    # Extract Metainfo from packaged Landsat Data
    tar = tarfile.open(inFile)
    # members = tar.getmembers()

    # Name of current Landsat file, extracted from tarfile name
    LSname = inFile[inFile.rfind("/") + 1:inFile.find(".")]

    # Define output Folder and extract tar-file into it
    curFol = outFol + LSname + "/"
    funcs.chkdir2(curFol)
    tar.extractall(curFol)

    # Read out the metafile
    metaDict = metaData(curFol, LSname)

    # collect filenames in a list to maybe delete later
    orgList = []
    for x in os.listdir(curFol):
        orgList.append(curFol + x)

    # Fill dictionary with radiation and toa
    rrdict = rad_toa(LSname, curFol, metaDict)

    # update rrdict if Temperatures should be calculated
    if calcKelvin > 0:
        kDict = cKelvin(calcKelvin, LSname, curFol, metaDict)
        rrdict.update(kDict)

    # Save radioation rasters to disk. Thermal bands only if temperature calculated before
    if saveRadiation > 0:
        saveRadFunc(LSname, curFol, rrdict, calcKelvin)

    if allIndices > 0:
        specDict = specIndFunc(LSname, curFol, allIndices, rrdict, saveInd)
        rrdict.update(specDict)

    if saveReflectance > 0:
        saveReflFunc(LSname, curFol, rrdict)

    return rrdict

    # delete originally extracted files
    if deleteOriginal == 2:
        for x in orgList:
            if x[-4] == ".":
                os.remove(x)
            else:
                shutil.rmtree(x)  # accounts for folder, os.remove only removes files
    elif deleteOriginal == 3:
        for x in orgList:
            if x[-7:].lower() != "mtl.txt":
                if x[-4] == ".":
                    os.remove(x)
                else:
                    shutil.rmtree(x)  # accounts for folder, os.remove only removes files


if __name__ == "__main__":
    inFol = r"F:\cali_spatial_vari\test/"  # containing untouched *.tar.gz files
    outFol = r"F:\cali_spatial_vari\test/"  # in which a sub-folder for each scene is created

    deleteOriginal = 1  # 1 = keep all, 2 = delete all, 3 = delete but keep metafile
    # Note: folder needs to be empty at start, else all is deleted
    saveRadiation = 0  # if True Radiation Tiffs will be saved
    saveReflectance = 1  # if True Reflectance Tiffs will be saved
    calcKelvin = 0  # 0 = no calculation, 1 = calc Band 10&11 or 61/62,
    # 2 = calc B10/61 only, 3=calc B11/62 only
    allIndices =0  # 0 = none, 1= all, 2 = NDVI, 3 = EVI, 4 = SAVI,
    # 5 = MSAVI, 6 = NDMI, 7 = NBR, 8 = NBR2, 9 = NDSI
    saveInd = 0  # save Indices to disk? 0 = no, 1 = yes (default)

    fileList = []
    for files in os.listdir(inFol):
        fileList.append(inFol + files)

    for inFilex in fileList:
        myDict = Landsat(inFilex, outFol, deleteOriginal, saveRadiation,
                         saveReflectance, calcKelvin, allIndices)

        myDict = {}