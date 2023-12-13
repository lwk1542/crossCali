# standard imports
import os

from landsat_metadata import landsat_metadata

# from dnppy import core


__all__ = ['toa_radiance_8',  # complete
           'toa_radiance_457']  # complete


def create_outname(outdir, inname, suffix, ext=False):
    """
    Quick way to create unique output file names within iterative functions

    This function is built to simplify the creation of output file names. Function allows
    ``outdir = False`` and will create an outname in the same directory as inname. Function will
    add a the user input suffix, separated by an underscore "_" to generate an output name.
    this is useful when performing a small modification to a file and saving new output with
    a new suffix. Function merely returns an output name, it does not save the file as that name.

    :param outdir:      either the directory of the desired outname or False to create an outname
                        in the same directory as the inname
    :param inname:      the input file from which to generate the output name "outname"
    :param suffix:      suffix to attach to the end of the filename to mark it as output
    :param ext:         specify the file extension of the output filename. Leave blank or False
                        and the outname will inherit the same extension as inname.

    :return outname:    the full filepath at which a new file can be created.

    """

    # isolate the filename from its directory and extension
    if os.path.isfile(inname):
        head, tail = os.path.split(inname)
        noext = tail.split('.')[:-1]
        noext = '.'.join(noext)
    else:
        head = ""
        tail = inname
        if "." in inname:
            noext = tail.split('.')[:-1]
            noext = '.'.join(noext)
        else:
            noext = inname

    # create the suffix
    if ext:
        suffix = "_{0}.{1}".format(suffix, ext)
    else:
        ext = tail.split('.')[-1:]
        suffix = "_{0}.{1}".format(suffix, ''.join(ext))

    if outdir:
        outname = os.path.join(outdir, noext + suffix)
        return outname
    else:
        outname = os.path.join(head, noext + suffix)
        return outname


def enf_list(item):
    """
    When a list is expected, this function can be used to ensure
    non-list data types are placed inside of a single entry list.

    :param item:    any datatype
    :return list:   a list type
    """

    if not isinstance(item, list) and item:
        return [item]
    else:
        return item


def toa_radiance_8(band_nums, meta_path, outdir=None):
    """
    Top of Atmosphere radiance (in Watts/(square meter x steradians x micrometers))
    conversion for landsat 8 data. To be performed on raw Landsat 8
    level 1 data. See link below for details:
    see here http://landsat.usgs.gov/Landsat8_Using_Product.php

    :param band_nums:   A list of desired band numbers such as [3, 4, 5]
    :param meta_path:   The full filepath to the metadata file for those bands
    :param outdir:      Output directory to save converted files.

    :return output_filelist:    List of filepaths created by this function.
    """

    meta_path = os.path.abspath(meta_path)
    output_filelist = []

    # enforce list of band numbers and grab the metadata from the MTL file
    band_nums = enf_list(band_nums)
    band_nums = map(str, band_nums)
    meta = landsat_metadata(meta_path)

    OLI_bands = ['1', '2', '3', '4', '5', '6', '7', '8', '9']

    # loop through each band
    for band_num in band_nums:
        if band_num in OLI_bands:
            # create the band name
            band_path = meta_path.replace("MTL.txt", "B{0}.tif".format(band_num))
            Qcal = arcpy.Raster(band_path)

            null_raster = arcpy.sa.SetNull(Qcal, Qcal, "VALUE = 0")

            # scrape the attribute data
            Ml = getattr(meta, "RADIANCE_MULT_BAND_{0}".format(band_num))  # multiplicative scaling factor
            Al = getattr(meta, "RADIANCE_ADD_BAND_{0}".format(band_num))  # additive rescaling factor

            # calculate Top-of-Atmosphere radiance
            TOA_rad = (null_raster * Ml) + Al
            del null_raster

            # create the output name and save the TOA radiance tiff
            if "\\" in meta_path:
                name = meta_path.split("\\")[-1]
            elif "//" in meta_path:
                name = meta_path.split("//")[-1]

            rad_name = name.replace("_MTL.txt", "_B{0}".format(band_num))

            if outdir is not None:
                outdir = os.path.abspath(outdir)
                outname = create_outname(outdir, rad_name, "TOA_Rad", "tif")
            else:
                folder = os.path.split(meta_path)[0]
                outname = create_outname(folder, rad_name, "TOA_Rad", "tif")

            TOA_rad.save(outname)
            output_filelist.append(outname)
            print("Saved toa_radiance at {0}".format(outname))
        # if listed band is not a OLI sensor band, skip it and print message
        else:
            print("Can only perform reflectance conversion on OLI sensor bands")
            print("Skipping band {0}".format(band_num))

    return output_filelist