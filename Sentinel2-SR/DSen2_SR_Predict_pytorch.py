import argparse
import os
import re
import sys
from osgeo import gdal
gdal.UseExceptions()  # Enable exceptions explicitly

from collections import defaultdict
import scipy.ndimage
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchmetrics.image import ErrorRelativeGlobalDimensionlessSynthesis, PeakSignalNoiseRatio, SpectralAngleMapper

torch.manual_seed(0)

import logging
logging.disable(logging.WARNING)


class ResidualBlock(nn.Module):
    def __init__(self, channels, dropout, scale=0.1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout(dropout)
        self.scale = scale

    def forward(self, x):
        residual = x
        out = torch.relu(self.conv1(x))
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.dropout1(out)
        out *= self.scale
        out += residual
        return out


class DSen2Net_2x(nn.Module):
    def __init__(self, input_channels=10, output_channels=6, num_resblocks=6, feature_size=128, scale=0.1, dropout = 0.2):
        super(DSen2Net_2x, self).__init__()
        self.initial_conv = nn.Conv2d(input_channels, feature_size, kernel_size=3, padding=1)
        self.res_blocks = nn.Sequential(*[ResidualBlock(feature_size, dropout, scale) for _ in range(num_resblocks)])
        self.final_conv = nn.Conv2d(feature_size, output_channels, kernel_size=3, padding=1)  # Fix output channels
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, x10, x20):
        residual = x20
        x = torch.cat([x10, x20], dim=1)
        x = torch.relu(self.initial_conv(x))
        x = self.res_blocks(x)
        x = self.final_conv(x)
        #x = self.dropout1(x)
        x += residual
        return x

class DSen2Net_6x(nn.Module):
    def __init__(self, input_channels=12, output_channels=2, num_resblocks=6, feature_size=128, scale=0.1, dropout = 0.2):
        super(DSen2Net_6x, self).__init__()
        self.initial_conv = nn.Conv2d(input_channels, feature_size, kernel_size=3, padding=1)
        self.res_blocks = nn.Sequential(*[ResidualBlock(feature_size, dropout, scale) for _ in range(num_resblocks)])
        self.final_conv = nn.Conv2d(feature_size, output_channels, kernel_size=3, padding=1)  # Fix output channels
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, x10, x20, x60):
        residual = x60
        x = torch.cat([x10, x20, x60], dim=1)
        x = torch.relu(self.initial_conv(x))
        x = self.res_blocks(x)
        x = self.final_conv(x)
        #x = self.dropout1(x)
        x += residual
        return x

def get_band_short_name(description):
    if ',' in description:
        return description[:description.find(',')]
    if ' ' in description:
        return description[:description.find(' ')]
    return description[:3]

def validate_description(description):
    m = re.match("(.*?), central wavelength (\d+) nm", description)
    if m:
        return m.group(1) + " (" + m.group(2) + " nm)"

    return description

def mc_dropout_prediction(model, inputs, num_samples, scale = '2x'):

    model.train()

    if scale == '2x':
        preds = torch.stack([model(inputs[0], inputs[1]) for _ in range(num_samples)])

    if scale == '6x':
        preds = torch.stack([model(inputs[0], inputs[1], inputs[2]) for _ in range(num_samples)])

    mean_prediction = preds.mean(dim = 0)
    uncertainty = preds.std(dim=0)

    return mean_prediction, uncertainty

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description = "Perform super-resolution of Sentinel-2 with DSen2",
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', action = 'store', dest = 'input_file', help = "An input Sentinel-2 data file")

    args = parser.parse_args()

    output_file = 'SR_' + os.path.split(args.input_file)[-1].split('.')[-3] + '.tiff'
    print(output_file)

    output_uncertainty_file = 'SR_unc_' + os.path.split(args.input_file)[-1].split('.')[-3] + '.tiff'

    select_bands = 'B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B11,B12'

    select_bands = [x for x in re.split(',', select_bands)]

    raster = gdal.Open(args.input_file)

    datasets = raster.GetSubDatasets()
    tenMsets = []
    twentyMsets = []
    sixtyMsets = []
    unknownMsets = []
    for (dsname, dsdesc) in datasets:
        if '10m resolution' in dsdesc:
            tenMsets += [dsname, dsdesc]
        elif '20m resolution' in dsdesc:
            twentyMsets += [dsname, dsdesc]
        elif '60m resolution' in dsdesc:
            sixtyMsets += [dsname, dsdesc]
        else:
            unknownMsets += [dsname, dsdesc]

    validated_10m_bands = []
    validated_10m_indices = []
    validated_20m_bands = []
    validated_20m_indices = []
    validated_60m_bands = []
    validated_60m_indices = []
    validated_descriptions = defaultdict(str)
    validated_descriptions_unc = defaultdict(str)

    sys.stdout.write("Selected 10m bands:")
    ds10 = gdal.Open(tenMsets[0])
    for b in range(0, ds10.RasterCount):
        desc = validate_description(ds10.GetRasterBand(b+1).GetDescription())
        shortname = get_band_short_name(ds10.GetRasterBand(b+1).GetDescription())
        if shortname in select_bands:
            sys.stdout.write(" " + shortname)
            select_bands.remove(shortname)
            validated_10m_bands += [shortname]
            validated_10m_indices += [b]
            validated_descriptions[shortname] = desc

    sys.stdout.write("\nSelected 20m bands:")
    ds20 = gdal.Open(twentyMsets[0])
    for b in range(0, ds20.RasterCount):
        desc = validate_description(ds20.GetRasterBand(b+1).GetDescription())
        shortname = get_band_short_name(ds20.GetRasterBand(b+1).GetDescription())
        if shortname in select_bands:
            sys.stdout.write(" " + shortname)
            select_bands.remove(shortname)
            validated_20m_bands += [shortname]
            validated_20m_indices += [b]
            validated_descriptions[shortname] = desc
            validated_descriptions_unc[shortname] = desc

    sys.stdout.write("\nSelected 60m bands:")
    ds60 = gdal.Open(sixtyMsets[0])
    for b in range(0, ds60.RasterCount):
        desc = validate_description(ds60.GetRasterBand(b + 1).GetDescription())
        shortname = get_band_short_name(ds60.GetRasterBand(b + 1).GetDescription())
        if shortname in select_bands:
            sys.stdout.write(" " + shortname)
            select_bands.remove(shortname)
            validated_60m_bands += [shortname]
            validated_60m_indices += [b]
            validated_descriptions[shortname] = desc
            validated_descriptions_unc[shortname] = desc

    sys.stdout.write("\n")

    driver = gdal.GetDriverByName('GTiff')
    outdata = driver.Create(output_file, ds10.RasterXSize, ds10.RasterXSize, 12, gdal.GDT_UInt16)
    outdata.SetGeoTransform(ds10.GetGeoTransform())
    outdata.SetProjection(ds10.GetProjection())

    outdata_unc = driver.Create(output_uncertainty_file, ds10.RasterXSize, ds10.RasterXSize, 8, gdal.GDT_Float32)
    outdata_unc.SetGeoTransform(ds10.GetGeoTransform())
    outdata_unc.SetProjection(ds10.GetProjection())


    for i, (k, v) in enumerate(validated_descriptions.items()):
        RasterBand = outdata.GetRasterBand(i+1)
        RasterBand.SetDescription(v)

    for i, (k, v) in enumerate(validated_descriptions_unc.items()):
        RasterBand = outdata_unc.GetRasterBand(i+1)
        RasterBand.SetDescription(v)



    model_2x = DSen2Net_2x()
    x2_path = os.path.join('Training', 'Model_2x_lr0.0001_bs64_epochs150_2025-05-08_23:07:52', 'DSEN2_2x_model.pth')
    checkpoint = torch.load(x2_path, map_location=device, weights_only=True)
    model_2x.load_state_dict(checkpoint['model_state_dict'])
    model_2x.to(device)

    model_6x = DSen2Net_6x()
    x6_path = os.path.join('Training', 'Model_6x_lr0.0001_bs64_epochs150_2025-05-08_10:54:34', 'DSEN2_x6_model.pth')
    checkpoint = torch.load(x6_path, map_location=device, weights_only=True)
    model_6x.load_state_dict(checkpoint['model_state_dict'])
    model_6x.to(device)

    chunk_size = 20 # Based on 60 m

    range_i = np.arange(0, ds60.RasterYSize // chunk_size) * chunk_size
    if not(np.mod(ds60.RasterYSize, chunk_size) == 0):
        range_i = np.append(range_i, ds60.RasterYSize - chunk_size)


    for i in tqdm(range_i):
        data10 = ds10.ReadAsArray(xoff = 0, yoff = int(i * 6), xsize = ds10.RasterXSize, ysize = 6 * chunk_size)[validated_10m_indices, :, :]

        data20 = ds20.ReadAsArray(xoff = 0, yoff = int(i * 3), xsize = ds20.RasterXSize, ysize = 3 * chunk_size)[validated_20m_indices, :, :]

        data60 = ds60.ReadAsArray(xoff = 0, yoff = int(i), xsize = ds60.RasterXSize, ysize = chunk_size)[validated_60m_indices, :, :]

        data20_ = np.zeros([data20.shape[0], data20.shape[1]*2, data20.shape[2]*2])
        for k in range(data20.shape[0]):
            data20_[k] = scipy.ndimage.zoom(data20[k], 2, order=1)


        data60_ = np.zeros([data60.shape[0], data60.shape[1]*6, data60.shape[2]*6])
        for l in range(data60.shape[0]):
            data60_[l] = scipy.ndimage.zoom(data60[l], 6, order=1)


        data10 = (data10 / 2000.).astype(np.float32)
        data20_ = (data20_ / 2000.).astype(np.float32)
        data60_ = (data60_ / 2000.).astype(np.float32)

        X10 = torch.from_numpy(data10).unsqueeze(0).to(device)
        X20 = torch.from_numpy(data20_).unsqueeze(0).to(device)
        X60 = torch.from_numpy(data60_).unsqueeze(0).to(device)

        with torch.no_grad():
            test_2X, uncertainty_2X = mc_dropout_prediction(model_2x, [X10, X20], 30, '2x')
            test_6X, uncertainty_6X = mc_dropout_prediction(model_6x, [X10, X20, X60], 30, '6x')
            data10  *= 2000
            test_2X *= 2000
            test_6X *= 2000

            test_2X = np.squeeze(test_2X, axis = 0)
            uncertainty_2X = np.squeeze(uncertainty_2X, axis = 0)
            test_6X = np.squeeze(test_6X, axis = 0)
            uncertainty_6X = np.squeeze(uncertainty_6X, axis = 0)

            total = np.concatenate([data10, test_2X.cpu().numpy(), test_6X.cpu().numpy()], axis = 0)

            outdata.WriteArray(total, xoff = 0, yoff = int(i * 6))
            outdata.FlushCache()

            uncertainty_total = np.concatenate([uncertainty_2X.cpu().numpy(), uncertainty_6X.cpu().numpy()], axis = 0)

            outdata_unc.WriteArray(uncertainty_total, xoff = 0, yoff = int(i * 6))
            outdata_unc.FlushCache()



    data10 = None
    data20 = None
    data60 = None
    outdata = None
    ds10 = None
    ds20 = None
    ds60 = None
