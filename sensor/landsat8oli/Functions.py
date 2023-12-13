# -*- coding: utf-8 -*-

import subprocess

import datetime
import gdal
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import os
import osr
import random
import scipy
import scipy.ndimage
import scipy.stats
import string
import xlrd
from gdalconst import *
# import UTMconversion as UTMconv

#########################################################################################
# Test if folder exists and create if not
#########################################################################################

# if curDir already exists, delete it and recreate it
def chkdir(curDir):
    if os.path.isdir(curDir):
        try:
            os.remove(curDir)
        except:
            print("Folder exists but cannot be deleted")
    try:
        os.mkdir(curDir)
    except:
        print("Folder already exists. Content will probably be overwritten")

    return curDir


# if curDir already exists, do nothing, else create it
def chkdir2(curDir):
    if os.path.isdir(curDir):
        return
    else:
        try:
            os.mkdir(curDir)
        except:
            print("Cannot create Folder: ", curDir)

    return curDir


#########################################################################################
# Get largest value from several arrays
#########################################################################################

# Return from several arrays one array with the respective largest value at
# each position


def getLargVal(*inA):
    inputlen = len(inA)

    for i in range(inputlen):
        if i == 0:
            condlist = [inA[0] > inA[1]]
            choicelist = [inA[0]]
            result = np.select(condlist, choicelist, inA[1])
        elif i == 1:
            continue
        else:
            condlist = [result > inA[i]]
            choicelist = [result]
            result = np.select(condlist, choicelist, inA[i])

    return result


# DEPRECATED: ONLY WORKS FOR UP TO 4 INPUT ARRAYS, OLD MANUAL WAY
def getLargVal_man(*inA):
    inputlen = len(inA)

    if inputlen == 2:
        condlist = [inA[0] > inA[1]]
        choicelist = [inA[0]]
        result = np.select(condlist, choicelist, inA[1])

    elif inputlen == 3:
        condlist = [np.logical_and(inA[0] > inA[1], inA[0] > inA[2]),
                    inA[1] > inA[2]]
        choicelist = [inA[0], inA[1]]
        result = np.select(condlist, choicelist, inA[2])


    elif inputlen == 4:
        condlist = [np.logical_and(inA[0] > inA[1],
                                   np.logical_and(inA[0] > inA[2], inA[0] > inA[3])),
                    np.logical_and(inA[1] > inA[2], inA[1] > inA[3]),
                    inA[2] > inA[3]]
        choicelist = [inA[0], inA[1], inA[2]]
        result = np.select(condlist, choicelist, inA[3])

    else:
        print("Only up to 4 arrays supported")

    return result


#########################################################################################
# Array to Raster conversion, two ways
#########################################################################################


def array_to_raster(inTiff, array, outFile, dataType=gdal.GDT_Float32):
    """
    Save a raster from a C order array. Standard output is GeoTiff.
    The attributes of an exisiting raster are used for the new output raster    
    
    Changed after the original
    http://gis.stackexchange.com/questions/58517
    /python-gdal-save-array-as-raster-with-projection-from-other-file
    
    inTiff   is an exisiting Tiff file, the attributes from this file are used
             to create the new one
    array    is the array that should be saved as a tiff   
    outFile  is the path and name of the desired output tiff
    
    """

    inDataset = gdal.Open(inTiff, GA_ReadOnly)

    # You need to get those values like you did.
    x_pixels = inDataset.RasterXSize  # number of pixels in x
    y_pixels = inDataset.RasterYSize  # number of pixels in y
    PIXEL_SIZE = inDataset.GetGeoTransform()[1]  # size of the pixel...
    x_min = inDataset.GetGeoTransform()[0]
    y_max = inDataset.GetGeoTransform()[3]  # x_min & y_max are like the "top left" corner.
    wkt_projection = inDataset.GetProjectionRef()

    driver = gdal.GetDriverByName('GTiff')

    outDataset = driver.Create(
        outFile,
        x_pixels,
        y_pixels,
        1,
        dataType, )

    outDataset.SetGeoTransform((
        x_min,  # 0
        PIXEL_SIZE,  # 1
        0,  # 2
        y_max,  # 3
        0,  # 4
        -PIXEL_SIZE))

    outDataset.SetProjection(wkt_projection)
    outDataset.GetRasterBand(1).WriteArray(array)
    outDataset.FlushCache()  # Write to disk.
    return outDataset, outDataset.GetRasterBand(
        1)  # If you need to return, remenber to return  also the dataset because the band don`t live without dataset.


# same as array_to_raster, but size definitions are given manually and not by inTiff
def array_to_raster_noTi(x_pix, y_pix, pixSize, x_min, y_max, proj, array, outFile):
    """Array > Raster
    Save a raster from a C order array.
    :param array: ndarray
    
    Changed after the original
    http://gis.stackexchange.com/questions/58517
    /python-gdal-save-array-as-raster-with-projection-from-other-file
    """

    driver = gdal.GetDriverByName('GTiff')

    outDataset = driver.Create(
        outFile,
        x_pix,
        y_pix,
        1,
        gdal.GDT_Float32, )

    outDataset.SetGeoTransform((
        x_min,  # 0 * top left border of pixel
        pixSize,  # 1
        0,  # 2
        y_max,  # 3 top left border of pixel
        0,  # 4
        -pixSize))

    projx = osr.SpatialReference()
    projx.SetWellKnownGeogCS(proj)  # Get the long coordinate system name
    wkt_projection = projx.ExportToWkt()

    outDataset.SetProjection(wkt_projection)
    outDataset.GetRasterBand(1).WriteArray(array)
    outDataset.FlushCache()  # Write to disk.
    return outDataset, outDataset.GetRasterBand(
        1)  # If you need to return, remenber to return  also the dataset because the band don`t live without dataset.


#########################################################################################
# LINEAR REGRESSION WITH TIME
#########################################################################################

# calculate linear regression and Mannn-Kendall-pValue coefficients for each raster coordinate
# returns a tuple of arrays

def linReg(inList):
    # Benchmark: a time series of 15 rasters, with each having 321x161 pix (51681), takes
    # about 19.1 seconds in total, with 5.8s for linReg and 13.2s for MK    

    # equally spaced time steps by length of inList
    timeList = np.asarray(list(range(len(inList))))
    stepLen = len(inList)

    # stack input arrays to make a 3D array
    dstack = np.dstack((inList))
    dstack1D = dstack.reshape(-1)

    # Break down dstack1D into a list, each element in list contains the single steps
    # of one pixel -> List length is equal to number of pixels
    # List can be used to use Pythons map() function
    dstackList = [dstack1D[i:i + stepLen] for i in range(0, len(dstack1D), stepLen)]

    # initialise empty arrays to be filled by output values, array are 1D
    slopeAr, intcptAr, rvalAr, pvalAr, stderrAr, mkPAr = [np.zeros(inList[0].reshape(-1).shape) for _ in range(6)]

    # Use map() to iterate over each pixels timestep values for linear reg and Mann.Kendall
    # Method is about 10% faster than using 2 for-loops (one for x- and y-axis)

    outListReg = list(map((lambda x: scipy.stats.linregress(timeList, x)), dstackList))

    for k in range(len(outListReg)):
        slopeAr[k] = outListReg[k][0]
        intcptAr[k] = outListReg[k][1]
        rvalAr[k] = outListReg[k][2]
        pvalAr[k] = outListReg[k][3]
        stderrAr[k] = outListReg[k][4]

    outListReg = []

    outListMK = list(map((lambda x: mk_test(x)), dstackList))
    for k in range(len(outListMK)):
        mkPAr[k] = outListMK[k][1]

    outShape = inList[0].shape
    outTuple = (slopeAr.reshape(outShape),
                intcptAr.reshape(outShape),
                rvalAr.reshape(outShape),
                pvalAr.reshape(outShape),
                stderrAr.reshape(outShape),
                mkPAr.reshape(outShape))

    return outTuple


#########################################################################################
# LINEAR REGRESSION BETWEEN TWO VARIABLES
#########################################################################################

"""

calculates the linear regression coefficients between two lists of tif rasters OR arrays,
though arrays must have the same dimensions, while Rasters may vary as described below:

If input are raster (tif) the entire path is needed in the input list

Input series can cover different extents. In this case, new rasters are created for 
 each series with the overlapping extent. 

If their resolution is different, the simpler one will be enlarged

If both stacks use different coordinate systems, the second stack is reprojected

outFol only needed for Raster input

"""


def linReg2(inList1, inList2, outFol=".../outFol/"):
    # test if both lists contain the same number of datasets/rasters
    if len(inList1) != len(inList2):
        print("Aborting! Input lists must contain same amount of datasets/rasters")
        return

    # Txt-file to log processing steps    
    infoTxt = open(outFol + "_Info.txt", "a")
    # First line is creation time and date
    nT = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
    infoTxt.write(("Process started [y-m-d]: " + nT + "\n" +
                   "Output Folder is: " + outFol + "\n\n"))

    iType = 0  # input type, 0 is default means inputs are arrays

    # If input is tif raster, convert to array before further use
    if type(inList1[0]) != np.ndarray:
        iType = 1
        refRas1 = inList1[0]
        refRas2 = inList2[0]

        firstRasGDAL1 = gdal.Open(refRas1, GA_ReadOnly)
        firstRasGDAL2 = gdal.Open(refRas2, GA_ReadOnly)

        proj1 = osr.SpatialReference()
        proj1.ImportFromWkt(firstRasGDAL1.GetProjectionRef())
        proj2 = osr.SpatialReference()
        proj2.ImportFromWkt(firstRasGDAL2.GetProjectionRef())

        infoTxt.write("Input: Raster Files \n"
                      "First Raster in First list: " + refRas1 +
                      "\t  Proj4: " + proj1.ExportToProj4() + "\n with cols/rows: " +
                      str(firstRasGDAL1.RasterXSize) + "/" +
                      str(firstRasGDAL1.RasterYSize) + "\n" +
                      "First Raster in Second list: " + refRas2 +
                      "\t  Proj4: " + proj2.ExportToProj4() + "\n with cols/rows: " +
                      str(firstRasGDAL2.RasterXSize) + "/" +
                      str(firstRasGDAL2.RasterYSize) + "\n\n")

        # Test if either projected or geographic coordinate systems is present
        if not ((proj1.IsProjected() or proj1.IsGeographic()) and \
                (proj2.IsProjected() or proj2.IsGeographic())):
            print("Coordinate System in one of the series is missing!")
            infoTxt.write("Aborted! Coordinate System in one of the series is missing!")
            infoTxt.close()
            return None

        # If raster and different coordinate systems and only partially overlap:
        if firstRasGDAL1.GetProjectionRef() != firstRasGDAL2.GetProjectionRef():
            intersection = my_intersect(refRas1, refRas2)

            infoTxt.write("Different Coordinate Systems -> Coordinate Systems changed to " +
                          proj1.ExportToProj4() + "\n" +
                          "Rasters intersect at [xmin ymin xmax ymax]: " +
                          str(intersection[0]) + " " + str(intersection[1]) + " " +
                          str(intersection[2]) + " " + str(intersection[3]) + "\n" +
                          "at " + outFol + "Scratch/ \n")

            # reset list1 and fill it with arrays, cropped to intersection
            nameList1 = []  # stores new pathnames
            list1 = []
            for dataset in inList1:
                newTif1 = reproject_dataset(dataset, refRas1, te=intersection, outFol=outFol + "Scratch/")
                nameList1.append(newTif1)
                for x in nameList1:
                    dataset = gdal.Open(x, GA_ReadOnly)
                    cols = firstRasGDAL1.RasterXSize
                    rows = firstRasGDAL1.RasterYSize
                    array = dataset.ReadAsArray(0, 0, cols, rows)
                    list1.append(array)
            infoTxt.write("First Rasters now have cols/rows: " + str(cols) + "/" +
                          str(rows) + "\n")

            nameList2 = []  # stores new pathnames
            list2 = []
            for dataset in inList2:
                newTif2 = reproject_dataset(dataset, refRas1, te=intersection, outFol=outFol + "Scratch/")
                nameList2.append(newTif2)

                # Define new first raster, as reprojection might have changed things
                nameList2FirstRas = nameList2[0]  # specs might have changed
                nameList2FirstGDAL = gdal.Open(nameList2FirstRas, GA_ReadOnly)

                for x in nameList2:
                    dataset = gdal.Open(x, GA_ReadOnly)
                    cols = nameList2FirstGDAL.RasterXSize
                    rows = nameList2FirstGDAL.RasterYSize
                    array = dataset.ReadAsArray(0, 0, cols, rows)
                    list2.append(array)

            infoTxt.write("Second Rasters now have cols/rows: " + str(cols) + "/" +
                          str(rows) + "\n\n"
                                      "New names of List1:   " + str(nameList1) + "\n",
                          "New names of List2:   " + str(nameList2) + "\n\n")




        # Raster share the same Coordinate System but not the same extent
        elif (firstRasGDAL1.GetProjectionRef() == firstRasGDAL2.GetProjectionRef()) and \
                (GetExtent(firstRasGDAL1) != GetExtent(firstRasGDAL2)):

            intersec = my_intersect(firstRasGDAL1, firstRasGDAL2)
            gdalTranslate = r'C:\Program Files (x86)\IDRISI Selva\GDAL\bin\gdal_translate.exe'

            infoTxt.write("ELIF linReg2 invoked: Rasters share the same" +
                          "Coordinate System but not the same extent \n" +
                          "Rasters intersect at [xmin ymin xmax ymax]: " +
                          str(intersec[0]) + " " + str(intersec[1]) + " " +
                          str(intersec[2]) + " " + str(intersec[3]) + "\n")

            newList1 = []
            newList2 = []

            # Create new files (on HDD) with shared extent from FIRST input list
            for file in inList1:
                chkdir2(outFol + "Cropped/")
                outPath = outFol + "Cropped/" + file[-13:]

                cmd = "-of GTiff -projwin " + str(intersec[0]) + " " + \
                      str(intersec[3]) + " " + \
                      str(intersec[2]) + " " + \
                      str(intersec[1]) + " "

                fullCmd = ' '.join([gdalTranslate, cmd, file, outPath])
                child = subprocess.Popen(fullCmd, stdout=subprocess.PIPE)
                child.wait()  # Wait for subprocess to finish, or pyhton continues and returns error when output is not there yet
                newList1.append(outPath)

            # Create new files (on HDD) with shared extent from SECOND input list
            for file in inList2:
                outPath = outFol + "Cropped/" + "2_" + file[-13:]

                cmd = "-of GTiff -projwin " + str(intersec[0]) + " " + \
                      str(intersec[3]) + " " + \
                      str(intersec[2]) + " " + \
                      str(intersec[1]) + " "

                fullCmd = ' '.join([gdalTranslate, cmd, file, outPath])
                child = subprocess.Popen(fullCmd, stdout=subprocess.PIPE)
                child.wait()
                newList2.append(outPath)

            # Read new created rasters in as arrays
            firstNewRas1 = newList1[0]
            firstNewRas2 = newList2[0]

            firstNewRasGDAL1 = gdal.Open(firstNewRas1, GA_ReadOnly)
            firstNewRasGDAL2 = gdal.Open(firstNewRas2, GA_ReadOnly)

            cols1 = firstNewRasGDAL1.RasterXSize
            rows1 = firstNewRasGDAL1.RasterYSize
            cols2 = firstNewRasGDAL2.RasterXSize
            rows2 = firstNewRasGDAL2.RasterYSize

            infoTxt.write("Raster cropped to shared extent and saved at " +
                          outFol + "Cropped/ \n" +
                          "First Rasters now have cols/rows: " +
                          str(cols1) + "/" + str(rows1) + "\n" +
                          "Second Rasters now have cols/rows: " +
                          str(cols2) + "/" + str(rows2) + "\n\n")

            list1 = []
            for file in newList1:
                dataset = gdal.Open(file, GA_ReadOnly)
                array = dataset.ReadAsArray(0, 0, cols1, rows1)
                list1.append(array)

            list2 = []
            for file in newList2:
                dataset = gdal.Open(file, GA_ReadOnly)
                array = dataset.ReadAsArray(0, 0, cols2, rows2)
                list2.append(array)



        # it is assumes that rasters may vary in cell size, but share the same extent
        else:
            # Open first rasters for meta information, done for both lists indiviually
            cols1 = firstRasGDAL1.RasterXSize
            rows1 = firstRasGDAL1.RasterYSize

            cols2 = firstRasGDAL2.RasterXSize
            rows2 = firstRasGDAL2.RasterYSize

            list1 = []  # reset list1 and fill it with arrays
            for x in inList1:
                dataset = gdal.Open(x, GA_ReadOnly)
                array = dataset.ReadAsArray(0, 0, cols1, rows1)
                list1.append(array)

            list2 = []  # reset list2 and fill it with arrays
            for x in inList2:
                dataset = gdal.Open(x, GA_ReadOnly)
                array = dataset.ReadAsArray(0, 0, cols2, rows2)
                list2.append(array)


    # If input are arrays, use them without any changes
    else:
        infoTxt.write("Input are arrays with cols/rows: " +
                      str(inList1[0].RasterXSize) + "/" + str(inList1[1].RasterYSize)
                      + "\n")
        list1 = inList1
        list2 = inList2

    list1a = list1
    list2a = list2

    # Test if dimensions of both input array series are the same. If not, resize the smaller
    # one with zoom function
    # http://stackoverflow.com/questions/13242382/resampling-a-numpy-array-representing-an-image
    size1 = list1[0].shape[0] * list1[0].shape[1]
    size2 = list2[0].shape[0] * list2[0].shape[1]

    if list1a[0].shape != list2a[0].shape:

        # set order for spline interpolation
        splOrder = 3

        infoTxt.write("Zoom function (scipy.ndimage.zoom) invoked because" +
                      "shape of arrays is list1 is " +
                      str(list1a[0].shape) + " and in list2 " + str(list2a[0].shape) + "\n")

        if size1 > size2:
            factorx = list1[0].shape[0] / list2[0].shape[0]
            factory = list1[0].shape[1] / list2[0].shape[1]
            list2a = []
            for x in list2:
                zoomArr = scipy.ndimage.zoom(x, (factorx, factory), order=splOrder)
                list2a.append(zoomArr)

            infoTxt.write("list2 arrays altered by factor for x and y: " +
                          str(factorx) + " " + str(factory) + "\n" +
                          "New shape of list2 arrays is " + str(list2a[0].shape) + "\n" +
                          "Order of spline interpolation was: " + str(splOrder) + "\n\n")

        else:
            factorx = list2[0].shape[0] / list1[0].shape[0]
            factory = list2[0].shape[1] / list1[0].shape[1]
            list1a = []
            for x in list1:
                zoomArr = scipy.ndimage.zoom(x, (factorx, factory), order=splOrder)
                list1a.append(zoomArr)

            infoTxt.write("list1 arrays altered by factor for x and y: " +
                          str(factorx) + " " + str(factory) + "\n" +
                          "New shape of list1 arrays is " + str(list1a[0].shape) + "\n" +
                          "Order of spline interpolation was: " + str(splOrder) + "\n\n")

    # initialise empty arrays to be filled by output values
    slopeAr, intcptAr, rvalAr, pvalAr, stderrAr = [np.zeros(list1a[0].shape) for _ in range(5)]

    xx = 1
    startTime = datetime.datetime.now()
    print("Process Starting")
    # iterate over every location in the arrays and calculate linreg
    for yCo in range(list1a[0].shape[0]):
        for xCo in range(list1a[0].shape[1]):

            valList1 = []
            for arrayX1 in list1a:
                valList1.append(arrayX1[yCo, xCo])

            valList2 = []
            for arrayX2 in list2a:
                valList2.append(arrayX2[yCo, xCo])

            slope, intcpt, rval, pval, stderr = scipy.stats.linregress(valList1, valList2)

            dPoints = list1a[0].shape[0] * list1a[0].shape[1]
            if xx % 100000 == 0:
                nowTime = datetime.datetime.now()

                # Overall time since start
                timedelta = nowTime - startTime
                timedeltaSec = timedelta.seconds
                timedeltaMin = timedeltaSec / 60
                allRate = xx / timedeltaSec

                print(xx, "/", dPoints, "done at average: %.2f /s " % allRate,
                      "Total runtime: %.2f min" % timedeltaMin)

            slopeAr[yCo, xCo] = slope
            intcptAr[yCo, xCo] = intcpt
            rvalAr[yCo, xCo] = rval
            pvalAr[yCo, xCo] = pval
            stderrAr[yCo, xCo] = stderr

            xx = xx + 1

    # Return output, if array input, return arrays, else tiffs
    if iType == 0:  # if input was array, return array... if was raster, return raster
        outTuple = (slopeAr, intcptAr, rvalAr, pvalAr, stderrAr)
        return outTuple

    else:
        # if cropped folder exisits, new rsater dimensions were created, use first cropped raster as master
        if os.path.isdir(outFol + "Cropped/"):
            firstNewRas = [outFol + "Cropped/" + x for x in os.listdir(outFol + "Cropped/")][0]
            array_to_raster(firstNewRas, slopeAr, outFol + "slope.tif")
            array_to_raster(firstNewRas, intcptAr, outFol + "intcpt.tif")
            array_to_raster(firstNewRas, rvalAr, outFol + "rval.tif")
            array_to_raster(firstNewRas, pvalAr, outFol + "pval.tif")
            array_to_raster(firstNewRas, stderrAr, outFol + "stderr.tif")

        elif size2 > size1:  # if one of the tifs was resized, the ouput must use the new dimensions
            array_to_raster(inList2[0], slopeAr, outFol + "slope.tif")
            array_to_raster(inList2[0], intcptAr, outFol + "intcpt.tif")
            array_to_raster(inList2[0], rvalAr, outFol + "rval.tif")
            array_to_raster(inList2[0], pvalAr, outFol + "pval.tif")
            array_to_raster(inList2[0], stderrAr, outFol + "stderr.tif")

        else:
            array_to_raster(inList1[0], slopeAr, outFol + "slope.tif")
            array_to_raster(inList1[0], intcptAr, outFol + "intcpt.tif")
            array_to_raster(inList1[0], rvalAr, outFol + "rval.tif")
            array_to_raster(inList1[0], pvalAr, outFol + "pval.tif")
            array_to_raster(inList1[0], stderrAr, outFol + "stderr.tif")

    infoTxt.write("Process finished [y-m-d]: " + nT + "\n" +
                  "Total number of locations: " + str(dPoints) +
                  " with " + str(len(inList1)) + " dimensions")
    infoTxt.close()
    print("Processing Done! See output File " + outFol + "_Info.txt   for Details")


#########################################################################################
# TIFF TO ARRAY
#########################################################################################

# read an entire folder of rasters as arrays and store them in a list, default raster
# format is tif, ifStatm is an optional if statement
def tiffToarray(inFol, ifStatm=True, printOut=False, inFormat="tif"):
    for allRasters in os.listdir(inFol):
        if allRasters[-3:] == "tif":
            firstRasStr = inFol + allRasters
            break
    firstRasGDAL = gdal.Open(firstRasStr, GA_ReadOnly)
    cols = firstRasGDAL.RasterXSize
    rows = firstRasGDAL.RasterYSize

    finList = []
    for files in os.listdir(inFol):
        if files[-3:] == inFormat:
            if printOut:
                print(files)
            fileIn = inFol + files
            dataset = gdal.Open(fileIn, GA_ReadOnly)
            array = dataset.ReadAsArray(0, 0, cols, rows)

            finList.append(array)

    return finList


# http://stackoverflow.com/questions/20343500/efficient-1d-linear-regression-for-each-element-of-3d-numpy-array
# http://stackoverflow.com/questions/19282429/regression-along-a-dimension-in-a-numpy-array?rq=1


#########################################################################################
# TIFF TO ARRAY SINGLE
#########################################################################################


# read one raster (path) as array and return array object
# format is tif
def singleTifToArray(inRas):
    firstRasGDAL = gdal.Open(inRas, GA_ReadOnly)
    cols = firstRasGDAL.RasterXSize
    rows = firstRasGDAL.RasterYSize

    dataset = gdal.Open(inRas, GA_ReadOnly)
    array = dataset.ReadAsArray(0, 0, cols, rows)

    return array


#########################################################################################
# MANN-KENDALL-TEST FOR TRENDS
#########################################################################################

# Mann-Kendall-Test
###Originally from: http://www.ambhas.com/codes/statlib.py
# Script changed, now 35x faster than original

def mk_test(x, alpha=0.05):
    n = len(x)

    # calculate S    
    listMa = np.matrix(x)  # convert input List to 1D matrix
    subMa = np.sign(listMa.T - listMa)  # calculate all possible differences in matrix
    # with itself and save only sign of difference (-1,0,1)
    s = np.sum(subMa[np.tril_indices(n, -1)])  # sum lower left triangle of matrix

    # calculate the unique data
    # return_counts=True returns a second array that is equivalent to tp in old version
    unique_x = np.unique(x, return_counts=True)
    g = len(unique_x[0])

    # calculate the var(s)
    if n == g:  # there is no tie
        var_s = (n * (n - 1) * (2 * n + 5)) / 18
    else:  # there are some ties in data
        tp = unique_x[1]
        var_s = (n * (n - 1) * (2 * n + 5) + np.sum(tp * (tp - 1) * (2 * tp + 5))) / 18

    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s == 0:
        z = 0
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)

    # calculate the p_value
    p = 2 * (1 - scipy.stats.norm.cdf(abs(z)))  # two tail test
    h = abs(z) > scipy.stats.norm.ppf(1 - alpha / 2)

    return h, p


#########################################################################################
# hdf to tiff
#########################################################################################

"""
convert hdf files to tiffs, 

subset is the number of the raster subset (0 is default)

slicing is the number of pixels that will be kept from original input [xmin,xmax,ymin,ymax]

(0) during conversion, scaling can be applied (e.g. from DN to NDVI, LAI etc.)

Script is still not fully automatic:
   (1) pixel size must be given manually
   (2) x_min and y_max coordinates must be given manually
   (3) activate to create coordinate system manually
   (4) activate to extract coordinate system automatically (disable (3) then)
   (5) change output DataType (allowed types: http://www.gdal.org/gdal_8h.html)
"""


def hdfTOtif(nameHDF, outFile, subset=0, slicing=[0, 0, 0, 0]):
    driver = gdal.GetDriverByName('hdf4')
    driver2 = gdal.GetDriverByName('Gtiff')
    driver.Register()
    driver2.Register()

    # open Dataset
    inDS = gdal.Open(nameHDF, GA_ReadOnly)

    # extract the subset to convert
    try:
        inHDF = gdal.Open(inDS.GetSubDatasets()[subset][0], GA_ReadOnly)
    except:
        inHDF = inDS

    # extract Projection
    # exProj = inHDF.GetProjectionRef() # (4)

    cols = inHDF.RasterXSize
    rows = inHDF.RasterYSize

    array = inHDF.ReadAsArray(0, 0, cols, rows)  # HDF to numpyArray

    # Define pixel ranges to slice original array
    if slicing[0] == 0 and slicing[1] == 0 and slicing[2] == 0 and slicing[3] == 0:
        x_Start = 0
        y_Start = 0
        x_End = cols
        y_End = rows
    else:
        x_Start = slicing[0]
        x_End = slicing[1]
        y_Start = slicing[2]
        y_End = slicing[3]

    # array = numpy.delete(array, np.s_[0:2500], axis=1)
    array = array[y_Start:y_End, x_Start:x_End]

    array = array * 0.0005  # (0) Use this for scaling

    PIXEL_SIZE = 0.0089285714  # (1)

    #  if slicing happens, xmin and ymax need updating
    x_min = 68.0 + x_Start * PIXEL_SIZE  # (2) 
    y_max = 55.0 - y_Start * PIXEL_SIZE  # (2)

    proj = osr.SpatialReference()  # (3)
    proj.SetWellKnownGeogCS("EPSG:32662")  # (3) Get the long coordinate system name
    # proj.SetUTM(48, True)                  # (3) Add UTM information, True = North
    wkt_projection = proj.ExportToWkt()  # (3) export both to Wkt

    x_pixels = array.shape[1]
    y_pixels = array.shape[0]

    outDataset = driver2.Create(
        outFile,
        x_pixels,
        y_pixels,
        1,
        # gdal.GDT_Int16,)
        # gdal.GDT_Byte,)     # (5) for 8-bit unsigned integer output
        gdal.GDT_Float32, )

    outDataset.SetGeoTransform((
        x_min,  # 0
        PIXEL_SIZE,  # 1
        0,  # 2
        y_max,  # 3
        0,  # 4
        -PIXEL_SIZE))

    outDataset.SetProjection(wkt_projection)  # (3)
    # outDataset.SetProjection(exProj)          # (4)
    outDataset.GetRasterBand(1).WriteArray(array)
    outDataset.FlushCache()  # Write to disk.


#########################################################################################
# BIL to Tiff
#########################################################################################

# convert BIL files to tiffs


def BILtoTIF(inBilPath, outTifPath):
    inBil = gdal.Open(inBilPath)  # InBIL
    driver = gdal.GetDriverByName('Gtiff')  # Output Driver
    outTif = driver.CreateCopy(outTifPath, inBil, 0)

    # Properly close the datasets to flush to disk
    inBil = None
    outTif = None


#########################################################################################
# Extract ROW data lists from Excel
#########################################################################################


# Read row data from Excel and returns it as a list
# Input Excel file, sheet number (beginning with 1)
# row0 is the actual row as displayed in Excel
# startC and endC are the respective columns using the Excel letters, end is included

def extXLS(file, sheetx, row0, startC, endC):
    row = row0 - 1  # Excel starts at 1, Python at 0
    sheet = sheetx - 1  # Excel starts at 1, Python at 0

    if len(startC) == 1:
        start = string.ascii_uppercase.index(startC)
    else:
        start = ((string.ascii_uppercase.index(startC[0]) + 1) * 26 +
                 string.ascii_uppercase.index(startC[1]))

    if len(endC) == 1:
        end = string.ascii_uppercase.index(endC) + 1
    else:
        end = ((string.ascii_uppercase.index(endC[0]) + 1) * 26 +
               string.ascii_uppercase.index(endC[1]) + 1)

    xlsFile = xlrd.open_workbook(file)  # Open Excel File

    if isinstance(sheet, int):  # if sheet is integer, sheet will be chosen by index
        xlsSheet = xlsFile.sheet_by_index(sheet)
    else:
        xlsSheet = xlsFile.sheet_by_name(sheet)  # Choose the sheet by name instead

    # data = xlsSheet.cell_value(5,5)
    dataRow = xlsSheet.row_values(row, start, end)
    return dataRow


#########################################################################################
# Corner Coordinates and Reprojection of Coordinates
#########################################################################################    
# http://gis.stackexchange.com/questions/57834/how-to-get-raster-corner-coordinates-using-python-gdal-bindings

def GetExtent(ds, cols=0, rows=0):
    ''' Return list of corner coordinates from a geotransform
        #ul ll lr ur

        @type gt:   C{tuple/list}
        @param gt: geotransform
        @type cols:   C{int}
        @param cols: number of columns in the dataset
        @type rows:   C{int}
        @param rows: number of rows in the dataset
        @rtype:    C{[float,...,float]}
        @return:   coordinates of each corner
    '''
    cols = ds.RasterXSize
    rows = ds.RasterYSize

    gt = ds.GetGeoTransform()

    ext = []
    xarr = [0, cols]
    yarr = [0, rows]

    for px in xarr:
        for py in yarr:
            x = gt[0] + (px * gt[1]) + (py * gt[2])
            y = gt[3] + (px * gt[4]) + (py * gt[5])
            ext.append([x, y])
            # print(x,y)
        yarr.reverse()
    return ext


#########################################################################################


def ReprojectCoords(coords, src_srs, tgt_srs):
    # http://www.samuelbosch.com/2009/05/projections-and-transformation-2.html
    # Coordinate System can be calculated e.g. using the following (equivalent for second raster)
    # rasWGS = "D:/Test/twoFol/reProj/2000_WGS84.tif"
    # gdWGS = gdal.Open(rasWGS, GA_ReadOnly)
    # src_srs = osr.SpatialReference()
    # src_srs.ImportFromWkt(gdWGS.GetProjectionRef())

    # define from which system to which the transformation is performed
    coordinate_transformation = osr.CoordinateTransformation(src_srs, tgt_srs)

    # test if list is nested -> several coordinate pairs or not
    # http://stackoverflow.com/questions/24180879/python-check-if-a-list-is-nested-or-not
    if any(isinstance(i, list) for i in coords):

        trans_coords = []
        for x in coords:
            yLong = coordinate_transformation.TransformPoint(x[0], x[1])
            yShort = yLong[:-1]  # yLOng contains 3rd agrument, z-value?! This is cut of
            yShortList = list(yShort)
            trans_coords.append(yShortList)  # new coords returned as nested lists

    else:
        longtrans_coords = coordinate_transformation.TransformPoint(coords[0], coords[1])
        trans_coords = longtrans_coords[:-1]

    return trans_coords


#########################################################################################


# Small Tool to import the Spatial Reference from an EPSG Code

def get_spatialref(epsg_code):
    spatialref = osr.SpatialReference()
    spatialref.ImportFromEPSG(epsg_code)
    return spatialref


# Create a random text string
def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


#########################################################################################
# Calculate new extent from intersect of two rasters
#########################################################################################   
# http://gis.stackexchange.com/questions/16834/how-to-add-different-sized-rasters-in-gdal-so-the-result-is-only-in-the-intersec

# Input can be gdal datasets or strings to tif files
# Returns 4 coordinate pairs. Coordinate systems must match! If not input files must be a tif

# input1 would be reprojected

def my_intersect(input1, input2):
    # test if input is a string (filepath) or already gdal Dataset
    if type(input1) == str:
        gd1 = gdal.Open(input1, GA_ReadOnly)
    else:
        gd1 = input1

    if type(input2) == str:
        gd2 = gdal.Open(input2, GA_ReadOnly)
    else:
        gd2 = input2

    # Extract projection information of both datasets
    proj1 = osr.SpatialReference()
    proj1.ImportFromWkt(gd1.GetProjectionRef())
    proj2 = osr.SpatialReference()
    proj2.ImportFromWkt(gd2.GetProjectionRef())

    if (gd1.GetProjectionRef() != gd2.GetProjectionRef()):
        projectedRas = reproject_dataset(input1, input2)
        gd1 = gdal.Open(projectedRas, GA_ReadOnly)
        # print("gd2 is now :", projectedRas)

        # transform = osr.CoordinateTransformation(proj2, proj1)
        # gd2.Transform(transform)

    gt1 = gd1.GetGeoTransform()
    gt2 = gd2.GetGeoTransform()

    # Calculate extents of both datasets
    r1 = [gt1[0], gt1[3], gt1[0] + (gt1[1] * gd1.RasterXSize), gt1[3] + (gt1[5] * gd1.RasterYSize)]
    r2 = [gt2[0], gt2[3], gt2[0] + (gt2[1] * gd2.RasterXSize), gt2[3] + (gt2[5] * gd2.RasterYSize)]

    # Return minimum shared extent
    intersection = [max(r1[0], r2[0]),  # xmin
                    max(r1[3], r2[3]),  # ymin
                    min(r1[2], r2[2]),  # xmax
                    min(r1[1], r2[1]),  # ymax
                    ]

    return intersection


#########################################################################################
# Reproject a raster
#########################################################################################

# Creates a pyhsical copy of the reprojected raster and return its path as a string

def reproject_dataset(inRas, newProjDS, te=0, name="x", outFol="D:/Test/Scratch/"):
    # Probably easier Pythonic Way:
    # http://stackoverflow.com/questions/10454316/how-to-project-and-resample-a-grid-to-match-another-grid-with-gdal-python/10538634#10538634

    # inRas is the Raster to be reprojected
    # newProjDS provides the new coordinate system (can be Dataset or Raster) 
    # te can be a list of integer bounding coordinates  xmin ymin xmax ymax
    # name is the name of the output file, if omitted random name is used
    # outFolFol needed for intermediate file creation

    # CHeck if output Folder exists and create if not
    chkdir2(outFol)

    # gdalWarp = r'C:/Program Files/QGIS Wien/bin/gdalwarp.exe'
    gdalWarp = r'C:\Program Files\GDAL\gdalwarp.exe'

    # Create a random file name for intermediate raster creation
    if name == "x":
        outFileName = outFol + "gdal_" + id_generator(3) + ".tif"
        print(outFileName, " current outFileName")
    else:
        outFileName = outFol + name + ".tif"
        print(outFileName, " current outFileName")

    if type(newProjDS) == str:
        ingd = gdal.Open(newProjDS, GA_ReadOnly)
    else:
        ingd = newProjDS

    if type(inRas) == str:
        oldRasDS = gdal.Open(inRas, GA_ReadOnly)
    else:
        oldRasDS = inRas

    inProj = osr.SpatialReference()
    inProj.ImportFromWkt(ingd.GetProjectionRef())
    inProjEPSG = "EPSG:" + str(inProj.GetAttrValue("AUTHORITY", 1))

    oldProj = osr.SpatialReference()
    oldProj.ImportFromWkt(oldRasDS.GetProjectionRef())
    oldProjEPSG = "EPSG:" + str(oldProj.GetAttrValue("AUTHORITY", 1))

    print("  inProjEPSG: ", inProjEPSG, "  OldProjEPSG: ", oldProjEPSG)

    # "-s_srs " + inProjEPSG +

    cmd = "-t_srs " + inProjEPSG + " -co TILED=YES -r cubic -dstnodata -3.40282346639e+038"

    # if no te value is given, no correction of extent is done, else new values are used     
    if te != 0:
        cmd = cmd + " -te " + str(te[0]) + " " + str(te[1]) + " " + str(te[2]) + " " + str(te[3])
    # print(te, " ", cmd)

    fullCmd = ' '.join([gdalWarp, cmd, inRas, outFileName])
    child = subprocess.Popen(fullCmd, stdout=subprocess.PIPE)
    child.wait()  # Wait for subprocess to finish, otherwise pyhton continues and returns error when output is not there yet
    return outFileName


# streamdata = child.communicate()[0]    #used to see subprocess output
# print(streamdata, "\n\n", fullCmd)


#########################################################################################
# Assign coordinate system to a TIFF file
#########################################################################################

# A new identical tif containing the information is created, the old one deleted and the
# new one renamed to the original name

def addCS(EPSG, inFile):
    # create in-dataset and output file location (to be changed to original later)
    inDS = gdal.Open(inFile)
    outFile = inFile[:-4] + "x.tif"

    # Get CS information, change to Wkt notation, assign to Dataset
    newCS = get_spatialref(EPSG)
    wkt_projection = newCS.ExportToWkt()
    inDS.SetProjection(wkt_projection)

    # Create new tiff with CS
    driver = gdal.GetDriverByName('GTiff')
    driver.CreateCopy(outFile, inDS, 0, ['TILED=YES'])
    inDS = None  # release dataset

    # Remove original input and rename second file with original name
    os.remove(inFile)
    os.rename(outFile, inFile)

    """
    
    outDataset.GetRasterBand(1).WriteArray(array)
    outDataset.FlushCache()  # Write to disk.
    """


#########################################################################################
# Return histogram of a raster ds
#########################################################################################

# file should be a raster or array
# -3.40282306e+38


def histo(input1, inBins=100, inRange=None, inNormed=False, inWeights=None, inDensity=None,
          NoDataValue=None, draw=True):
    # test if input is a string (filepath) or already numpy array
    if type(input1) == str:
        array = singleTifToArray(input1)
    else:
        array = input1

    array2 = ma.masked_values(array, NoDataValue)
    pixNum = array2.count()  # size on masked arrays would still return masked pixels

    # returns tuple of 2 arrays, one with unique raster values and of with corresponding count
    unique = np.unique(array2.compressed())  # array of unique values
    uniqueMax = unique.max()  # maximum value

    array1D = array2.compressed()

    # To draw histograms with values smaller one or more than 256 use 100 bins by default
    if uniqueMax <= 1 or uniqueMax >= 256:
        inBins = inBins
    else:
        inBins = range(1, int(uniqueMax) + 2)

    h = np.histogram(array2.compressed(), bins=inBins, normed=inNormed,
                     weights=inWeights, density=inDensity)

    # Returns graphic histogram by standard
    if draw:
        plt.hist(array1D, bins=inBins, range=inRange, normed=inNormed, weights=inWeights)

    # plt.show()

    print("Total number of pixel (noData included): ", array2.size,
          "(noData excluded): ", pixNum)

    # return total pix number, masked pix number, and tuple of histogram array
    return array2.size, pixNum, h


#########################################################################################
# Create a mask from vector and convert vectors to raster
#########################################################################################

'''adopted from http://www.machinalis.com/blog/python-for-geospatial-data-processing/'''


def create_mask_from_vector(vector_data_path, cols, rows, geo_transform,
                            projection, target_value=1):
    """Rasterize the given vector (wrapper for gdal.RasterizeLayer)."""
    data_source = gdal.OpenEx(vector_data_path, gdal.OF_VECTOR)
    layer = data_source.GetLayer(0)
    driver = gdal.GetDriverByName('MEM')  # In memory dataset
    target_ds = driver.Create('', cols, rows, 1, gdal.GDT_UInt16)
    target_ds.SetGeoTransform(geo_transform)
    target_ds.SetProjection(projection)
    gdal.RasterizeLayer(target_ds, [1], layer, burn_values=[target_value])
    return target_ds


def vectors_to_raster(file_paths, rows, cols, geo_transform, projection):
    """Rasterize the vectors in the given directory in a single image."""
    labeled_pixels = np.zeros((rows, cols))
    for i, path in enumerate(file_paths):
        label = i + 1
        ds = create_mask_from_vector(path, cols, rows, geo_transform,
                                     projection, target_value=label)
        band = ds.GetRasterBand(1)
        labeled_pixels += band.ReadAsArray()
        ds = None
    return labeled_pixels