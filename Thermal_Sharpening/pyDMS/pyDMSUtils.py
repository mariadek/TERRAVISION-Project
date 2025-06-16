# -*- coding: utf-8 -*-

import numpy as np
import os
from osgeo import gdal, ogr
import netCDF4
import shutil
import scipy.ndimage as ndi
from numba import njit, stencil



def mask_extractor(input, output, sr = False):

    raster = gdal.Open(input)

    datasets = raster.GetSubDatasets()


    tenMsets = []
    twentyMsets = []
    for (dsname, dsdesc) in datasets:
        if '10m resolution' in dsdesc:
            tenMsets += [dsname, dsdesc]
        elif '20m resolution' in dsdesc:
            twentyMsets += [dsname, dsdesc]

    ds20 = gdal.Open(twentyMsets[0])
    for b in range(0, ds20.RasterCount):
        if 'SCL, Scene Classification' in ds20.GetRasterBand(b+1).GetDescription():
            scl_band = b+1

    ds10 = gdal.Open(tenMsets[0])

    driver = gdal.GetDriverByName('GTiff')
    # RasterXSize - columns
    # RasterYSize - rows
    outdata = driver.Create(output, ds10.RasterXSize, ds10.RasterYSize, 1, gdal.GDT_UInt16)
    outdata.SetGeoTransform(ds10.GetGeoTransform())
    outdata.SetProjection(ds10.GetProjection())

    data20 = ds20.GetRasterBand(scl_band).ReadAsArray(0, 0, ds20.RasterXSize, ds20.RasterYSize, buf_xsize = ds10.RasterXSize, buf_ysize = ds10.RasterYSize)

    outdata.WriteArray(data20, xoff = 0, yoff = 0)
    outdata.FlushCache()


    outdata = None
    data20 = None
    ds10 = None
    ds20 = None

def s3_preprocessor(filename, highfile):

    work_dir = 'tmp'
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    # 1. Load lon and lat data for preparation of making VRTS
    geo = netCDF4.Dataset(os.path.join(filename, "geodetic_in.nc"), 'r')
    lon = geo['longitude_in'][:]
    lat = geo['latitude_in'][:]
    # get dims of raster
    ysize, xsize = lon.shape

    # Save this lon/lat to TIFFs and VRT
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(
        os.path.join(work_dir,"lon.tif"),
        xsize,
        ysize,
        1,
        gdal.GDT_Float32, )
    band = dataset.GetRasterBand(1).WriteArray(lon.data)
    dataset = None
    dataset = driver.Create(
        os.path.join(work_dir,"lat.tif"),
        xsize,
        ysize,
        1,
        gdal.GDT_Float32, )
    band = dataset.GetRasterBand(1).WriteArray(lat.data)
    dataset = None

    # *-- Construct and save VRTs --*
    lon_vrt = f"""<VRTDataset rasterXSize="{xsize}" rasterYSize="{ysize}">
      <SRS>GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9108"]],AUTHORITY["EPSG","4326"]]</SRS>
  <VRTRasterBand dataType="Float32" band="1">
    <SimpleSource>
      <SourceFilename relativeToVRT="1">lon.tif</SourceFilename>
      <SourceBand>1</SourceBand>
      <SourceProperties RasterXSize="{xsize}" RasterYSize="{ysize}" DataType="Float32" BlockXSize="256" BlockYSize="256" />
      <SrcRect xOff="0" yOff="0" xSize="{xsize}" ySize="{ysize}" />
      <DstRect xOff="0" yOff="0" xSize="{xsize}" ySize="{ysize}" />
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>"""

    lat_vrt = f"""<VRTDataset rasterXSize="{xsize}" rasterYSize="{ysize}">
      <SRS>GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9108"]],AUTHORITY["EPSG","4326"]]</SRS>
  <VRTRasterBand dataType="Float32" band="1">
    <SimpleSource>
      <SourceFilename relativeToVRT="1">lat.tif</SourceFilename>
      <SourceBand>1</SourceBand>
      <SourceProperties RasterXSize="{xsize}" RasterYSize="{ysize}" DataType="Float32" BlockXSize="256" BlockYSize="256" />
      <SrcRect xOff="0" yOff="0" xSize="{xsize}" ySize="{ysize}" />
      <DstRect xOff="0" yOff="0" xSize="{xsize}" ySize="{ysize}" />
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>"""

    # save VRT files now
    with open(os.path.join(work_dir,"lon.vrt"), "w") as text_file:
        text_file.write(lon_vrt)
    with open(os.path.join(work_dir,"lat.vrt"), "w") as text_file:
        text_file.write(lat_vrt)

    # Pull out the data we need and put into a .tif
    dataset = driver.Create(
        os.path.join(work_dir, "data.tif"),
        xsize,
        ysize,
        1,
        gdal.GDT_Float32, )

    s3_lst = netCDF4.Dataset(os.path.join(filename, 'LST_in.nc'))
    lst = s3_lst.variables['LST'][:]
    bnd = dataset.GetRasterBand(1).WriteArray(lst.data)
    dataset = None

    dataset = driver.Create(
        os.path.join(work_dir, "mask.tif"),
        xsize,
        ysize,
        1,
        gdal.GDT_Float32, )

    s3_flags = netCDF4.Dataset(os.path.join(filename, 'flags_in.nc'))
    s3_mask = s3_flags.variables['bayes_in'][:]
    dataset = None

    vrt_raster_band_tmpl = f"""<VRTRasterBand dataType="Float32" band="1">
    <SimpleSource>
      <SourceFilename relativeToVRT="1">data.tif</SourceFilename>
      <SourceBand>1</SourceBand>
      <SourceProperties RasterXSize="{xsize}" RasterYSize="{ysize}" DataType="Float32" BlockXSize="256" BlockYSize="256" />
      <SrcRect xOff="0" yOff="0" xSize="{xsize}" ySize="{ysize}" />
      <DstRect xOff="0" yOff="0" xSize="{xsize}" ySize="{ysize}" />
    </SimpleSource>
    </VRTRasterBand>
    """
    # and
    lon_file = os.path.join(work_dir,"lon.tif")
    lat_file = os.path.join(work_dir,"lat.tif")
    vrt_main_tmpl= f"""<VRTDataset rasterXSize="{xsize}" rasterYSize="{ysize}">
       <metadata domain="GEOLOCATION">
         <mdi key="X_DATASET">{lon_file}</mdi>
         <mdi key="X_BAND">1</mdi>
         <mdi key="Y_DATASET">{lat_file}</mdi>
         <mdi key="Y_BAND">1</mdi>
         <mdi key="PIXEL_OFFSET">0</mdi>
         <mdi key="LINE_OFFSET">0</mdi>
         <mdi key="PIXEL_STEP">1</mdi>
         <mdi key="LINE_STEP">1</mdi>
       </metadata>
           {vrt_raster_band_tmpl}
    </VRTDataset>"""

    # save it
    with open(os.path.join(work_dir, "data.vrt"), "w") as text_file:
        text_file.write(vrt_main_tmpl)

    # make a real geotiff now
    g = gdal.Warp(os.path.join(work_dir, "one.tif"), os.path.join(work_dir, "data.vrt"), dstSRS="EPSG:4326", geoloc=True, srcNodata = -32768)
    g = None

    vrt_raster_band_tmpl = f"""<VRTRasterBand dataType="Float32" band="1">
    <SimpleSource>
      <SourceFilename relativeToVRT="1">mask.tif</SourceFilename>
      <SourceBand>1</SourceBand>
      <SourceProperties RasterXSize="{xsize}" RasterYSize="{ysize}" DataType="Float32" BlockXSize="256" BlockYSize="256" />
      <SrcRect xOff="0" yOff="0" xSize="{xsize}" ySize="{ysize}" />
      <DstRect xOff="0" yOff="0" xSize="{xsize}" ySize="{ysize}" />
    </SimpleSource>
    </VRTRasterBand>
    """
    # and
    lon_file = os.path.join(work_dir,"lon.tif")
    lat_file = os.path.join(work_dir,"lat.tif")
    vrt_main_tmpl= f"""<VRTDataset rasterXSize="{xsize}" rasterYSize="{ysize}">
       <metadata domain="GEOLOCATION">
         <mdi key="X_DATASET">{lon_file}</mdi>
         <mdi key="X_BAND">1</mdi>
         <mdi key="Y_DATASET">{lat_file}</mdi>
         <mdi key="Y_BAND">1</mdi>
         <mdi key="PIXEL_OFFSET">0</mdi>
         <mdi key="LINE_OFFSET">0</mdi>
         <mdi key="PIXEL_STEP">1</mdi>
         <mdi key="LINE_STEP">1</mdi>
       </metadata>
           {vrt_raster_band_tmpl}
    </VRTDataset>"""

    # save it
    with open(os.path.join(work_dir, "mask.vrt"), "w") as text_file:
        text_file.write(vrt_main_tmpl)

    # make a real geotiff now
    g = gdal.Warp(os.path.join(work_dir, "one_mask.tif"), os.path.join(work_dir, "mask.vrt"), dstSRS="EPSG:4326", geoloc=True)
    g = None


    # Perform re-projection
    scene_hr = gdal.Open(highfile)
    geoTransform = scene_hr.GetGeoTransform()
    minx = geoTransform[0]
    maxy = geoTransform[3]
    maxx = minx + geoTransform[1] * scene_hr.RasterXSize
    miny = maxy + geoTransform[5] * scene_hr.RasterYSize
    # Re-project to WGS84 first -- not sure why but seem to need to
    co = 'COMPRESS=DEFLATE', 'INTERLEAVE=BAND', 'PREDICTOR=2'
    g = gdal.Warp(os.path.join(work_dir, "s3_slstr.tif"), os.path.join(work_dir,"one.tif"),
                  dstSRS=scene_hr.GetProjection(),  outputBounds=(minx, miny, maxx, maxy),
                  xRes= 1000, yRes=-1000, srcNodata = -32768,
                  creationOptions=co)
    g = gdal.Warp(os.path.join(work_dir, "s3_slstr_mask.tif"), os.path.join(work_dir,"one_mask.tif"),
                  dstSRS=scene_hr.GetProjection(),  outputBounds=(minx, miny, maxx, maxy),
                  xRes= 1000, yRes=-1000,
                  creationOptions=co)
    g = None


    outpath = os.path.join(os.path.split(filename)[0], "Subset_" + os.path.split(filename)[-1].split('.')[-2] + ".tiff")
    outpath_flag = os.path.join(os.path.split(filename)[0], "Subset_Flag_" + os.path.split(filename)[-1].split('.')[-2] + ".tiff")
    shutil.move(os.path.join(work_dir, "s3_slstr.tif"), outpath)
    shutil.move(os.path.join(work_dir, "s3_slstr_mask.tif"), outpath_flag)

    # Now tidy everything up.. eg delete tmp files
    # Gather directory contents
    contents = [os.path.join(work_dir, i) for i in os.listdir(work_dir)]
    # Iterate and remove each item in the appropriate manner
    [os.remove(i) if os.path.isfile(i) or os.path.islink(i) else shutil.rmtree(i) for i in contents]
    os.rmdir(work_dir)

def binomialSmoother(data):
    def filterFunction(footprint):
        weight = [1, 2, 1, 2, 4, 2, 1, 2, 1]
        # Don't smooth land and invalid pixels
        if np.isnan(footprint[4]):
            return footprint[4]

        footprintSum = 0
        weightSum = 0
        for i in range(len(weight)):
            # Don't use land and invalid pixels in smoothing of other pixels
            if not np.isnan(footprint[i]):
                footprintSum = footprintSum + weight[i] * footprint[i]
                weightSum = weightSum + weight[i]
        try:
            ans = footprintSum/weightSum
        except ZeroDivisionError:
            ans = footprint[4]
        return ans

    smoothedData = ndi.filters.generic_filter(data, filterFunction, 3)

    return smoothedData

@njit
def removeEdgeNaNs(a, i, j):
    values = np.array([a[i-1, j], a[i+1, j], a[i, j-1], a[i, j+1]])
    values = values[~np.isnan(values)]  # Remove NaN values manually
    if values.size == 0:
        return np.nan  # Return NaN if all elements are NaN
    return values.mean()  # Compute mean of non-NaN values
