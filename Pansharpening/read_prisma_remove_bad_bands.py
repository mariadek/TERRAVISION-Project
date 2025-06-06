import os
import h5py
import numpy as np
from osgeo import gdal, osr
from tqdm import tqdm

gdal.UseExceptions()


def generate_PAN(path_img: str, save_dir: str) -> None:
	base_name = os.path.splitext(os.path.basename(path_img))[0]
	dir_path = os.path.join(save_dir, base_name)
	os.makedirs(dir_path, exist_ok=True)

	with h5py.File(path_img, 'r') as h5f:
		pan_dn = np.array(h5f['HDFEOS/SWATHS/PRS_L2D_PCO/Data Fields/Cube'][()], dtype=np.uint16)

		scale_min = h5f.attrs['L2ScalePanMin']
		scale_max = h5f.attrs['L2ScalePanMax']
		pan_reflectance = scale_min + pan_dn * (scale_max - scale_min) / 65535.0

		geo = {
			'proj_code': h5f.attrs['Projection_Id'],
			'proj_name': h5f.attrs['Projection_Name'],
			'epsg': int(h5f.attrs['Epsg_Code']),
			'xmin': min(h5f.attrs['Product_ULcorner_easting'], h5f.attrs['Product_LLcorner_easting']),
			'xmax': max(h5f.attrs['Product_URcorner_easting'], h5f.attrs['Product_LRcorner_easting']),
			'ymin': min(h5f.attrs['Product_LLcorner_northing'], h5f.attrs['Product_LRcorner_northing']),
			'ymax': max(h5f.attrs['Product_ULcorner_northing'], h5f.attrs['Product_URcorner_northing']),
		}

	res = 5
	height, width = pan_reflectance.shape
	geotransform = (geo['xmin'], res, 0, geo['ymax'], 0, -res)

	srs = osr.SpatialReference()
	srs.ImportFromEPSG(geo['epsg'])
	wkt_proj = srs.ExportToWkt()

	output_path = os.path.join(dir_path, f"{base_name}_PAN.tiff")

	driver = gdal.GetDriverByName('GTiff')
	dataset = driver.Create(output_path, width, height, 1, gdal.GDT_Float32)
	dataset.SetGeoTransform(geotransform)
	dataset.SetProjection(wkt_proj)
	dataset.GetRasterBand(1).WriteArray(pan_reflectance)
	dataset.FlushCache()
	dataset = None

	print(f"Saved PAN image to: {output_path}")

def search_band_index(bands,cw,target_wavelength):
	if len(cw) != bands.shape[2]:
		raise ValueError('Dimensions do not match!')
	else:
		temp = np.abs(cw - target_wavelength)
		target_band_index = np.where(temp == temp.min())[0]
		#print(f'Index of target wavelength {target_wavelength} nm is {target_band_index}')
		return target_band_index
		#target_band = np.squeeze(bands[:,target_band_index,:])
		#return target_band

def count_defective_pixels(arr, threshold):
	# 0 flagged pixels are ok
	zero_counts = np.sum(arr == 0, axis=(0, 2))
	total_elements = arr.shape[0] * arr.shape[2]
	zero_percent = (zero_counts / total_elements) * 100

	#print(zero_percent)  # This gives a (173,) array with percentages for each band
	high_zero_indices = np.where(zero_percent < threshold)[0]
	return list(high_zero_indices)

def generate_HS(path_img: str, save_dir: str) -> None:
	base_name = os.path.splitext(os.path.basename(path_img))[0]
	dir_path = os.path.join(save_dir, base_name)
	os.makedirs(dir_path, exist_ok=True)

	with h5py.File(path_img, 'r') as h5f:
		geo = {'proj_code':h5f.attrs['Projection_Id'],
			   'proj_name':h5f.attrs['Projection_Name'],
			   'proj_epsg':h5f.attrs['Epsg_Code'],
			   'xmin':np.min([h5f.attrs['Product_ULcorner_easting'], h5f.attrs['Product_LLcorner_easting']]),
			   'xmax':np.max([h5f.attrs['Product_LRcorner_easting'], h5f.attrs['Product_URcorner_easting']]),
			   'ymin':np.min([h5f.attrs['Product_LLcorner_northing'], h5f.attrs['Product_LRcorner_northing']]),
			   'ymax':np.max([h5f.attrs['Product_ULcorner_northing'], h5f.attrs['Product_URcorner_northing']])}

		CW = np.concatenate([h5f.attrs['List_Cw_Vnir'][::-1], h5f.attrs['List_Cw_Swir'][::-1]])

		Flag = np.concatenate([h5f.attrs['CNM_VNIR_SELECT'][::-1], h5f.attrs['CNM_SWIR_SELECT'][::-1]])

		SWIR_bands = np.array(h5f['HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/SWIR_Cube'])
		VNIR_bands = np.array(h5f['HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/VNIR_Cube'])

		SWIR_bands_C = np.swapaxes(SWIR_bands, 1, 2)
		VNIR_bands_C = np.swapaxes(VNIR_bands, 1, 2)
		VNIR_bands_CC = VNIR_bands_C[:, :, ::-1]
		SWIR_bands_CC = SWIR_bands_C[:, :, ::-1]

		L2ScaleSwirMax = h5f.attrs['L2ScaleSwirMax']
		L2ScaleSwirMin = h5f.attrs['L2ScaleSwirMin']
		L2ScaleVnirMax = h5f.attrs['L2ScaleVnirMax']
		L2ScaleVnirMin = h5f.attrs['L2ScaleVnirMin']

		SWIR_bands_R = np.float32(SWIR_bands_CC.copy())
		for n in range(SWIR_bands_CC.shape[2]):
			SWIR_bands_R[:,:,n] = L2ScaleSwirMin + SWIR_bands_CC[:,:,n]*(L2ScaleSwirMax-L2ScaleSwirMin)/65535

		VNIR_bands_R = np.float32(VNIR_bands_CC.copy())
		for n in range(VNIR_bands_CC.shape[2]):
			VNIR_bands_R[:,:,n] = L2ScaleVnirMin + VNIR_bands_CC[:,:,n]*(L2ScaleVnirMax - L2ScaleVnirMin)/65535

		img = np.concatenate([VNIR_bands_R,SWIR_bands_R], axis=2)

		# Delete incorrect bands
		# Zero flagged
		zero_wavelength_indices_ALL = list(np.where(CW == 0.0)[0])
		zero_flagged_indices_ALL = list(np.where(Flag == 0)[0])

		wavelengths_vnir_water_vapour = list([920, 423.78476, 415.839, 406.9934])

		if len(CW) != img.shape[2]:
			raise ValueError('Dimensions do not match!')
		else:
			IMG_VNIR_water_vapour = {wavelength: search_band_index(img, CW, wavelength)[0]
						for wavelength in wavelengths_vnir_water_vapour}

		wavelengths_swir_water_vapour = list(range(1350, 1480, 10)) + list(range(1800, 1960, 10)) + list([1120, 2020]) + \
				list([2497.1155, 2490.2192, 2483.793 , 2477.055 , 2469.6272, 2463.0303, \
				  2456.5857, 2449.1423, 2442.403 , 2435.5442, 2428.6677, 2421.2373, \
				  2414.3567, 2407.6045, 2400.036 , 2393.0388, 2386.0618, 2378.771, \
				  2371.5522, 2364.5945, 2357.2937, 2349.7915, 2342.8228])

		if len(CW) != img.shape[2]:
			raise ValueError('Dimensions do not match!')
		else:
			IMG_SWIR_water_vapour = {wavelength: search_band_index(img, CW, wavelength)[0]
						for wavelength in wavelengths_swir_water_vapour}

		bands_to_remove = list(IMG_SWIR_water_vapour.values()) + list(IMG_VNIR_water_vapour.values())

		SWIR_pixel_error = np.array(h5f['HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/SWIR_PIXEL_L2_ERR_MATRIX'][()], dtype=np.uint16)[::-1]
		VNIR_pixel_error = np.array(h5f['HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/VNIR_PIXEL_L2_ERR_MATRIX'][()], dtype=np.uint16)[::-1]

		SWIR_zero_indices = count_defective_pixels(SWIR_pixel_error, 90)
		VNIR_zero_indices = count_defective_pixels(VNIR_pixel_error, 90)

		bands_to_remove_final = bands_to_remove + SWIR_zero_indices + VNIR_zero_indices + list(zero_flagged_indices_ALL) + list(zero_wavelength_indices_ALL)

		all_bands = np.arange(img.shape[2])  # Assuming shape is (H, W, Bands)

		bands_to_keep = np.setdiff1d(all_bands, bands_to_remove_final)

		ALL_filtered = img[:, :, bands_to_keep]

		res = 30

		ex = {'xmin' : geo['xmin'],
			  'xmax': geo['xmin'] + img.shape[1] * res,
			  'ymin': geo['ymin'],
			  'ymax': geo['ymin'] + img.shape[0] * res}

		# Set the resolution
		GeoT = (ex['xmin'], res, 0, ex['ymax'], 0, -res)

		driver = gdal.GetDriverByName('GTiff')
		Projj = osr.SpatialReference()
		Projj.ImportFromEPSG(int(geo['proj_epsg'])) #4326
		Projj.ExportToPrettyWkt()

		output_path = os.path.join(dir_path, f"{base_name}_HS.tiff")

		rows = img.shape[0]
		cols = img.shape[1]
		band_num = len(bands_to_keep)

		DataSet = driver.Create(output_path, cols, rows, band_num, gdal.GDT_Float32)
		DataSet.SetGeoTransform(GeoT)
		DataSet.SetProjection(Projj.ExportToWkt())

		for i, band in enumerate(tqdm(bands_to_keep, desc="Writing bands")):
			RasterBand = DataSet.GetRasterBand(i+1)
			RasterBand.SetDescription('Wavelength: ' + str(CW[band]))

			RasterBand.WriteArray(img[:, :, band])
			DataSet.FlushCache()


if __name__ == '__main__':

	PRISMA_dir = 'PRS_L2D_STD_20200627102358_20200627102402_0001.he5' # Add the path of the PRISMA .he5 file after unzipping

	generate_PAN(PRISMA_dir, 'PRISMA_Datacubes')
	generate_HS(PRISMA_dir, 'PRISMA_Datacubes')
