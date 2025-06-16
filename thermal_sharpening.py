import argparse
import os
import time

from zipfile import ZipFile
from osgeo import gdal
import numpy as np

from Sentinel2SR.Sentinel2SR import run_Sentinel2_SR, run_Sentinel2_resampling

import pyDMS.pyDMSUtils as utils
from pyDMS.pyDMS import DecisionTreeSharpener, NeuralNetworkSharpener
from pyDMS.pyDMS import REG_sknn_ann, REG_sklearn_ann

if __name__ == "__main__":

    '''
    parser = argparse.ArgumentParser(description = "Data Mining for Sharpening Sentinel 3 SLSTR with Sentinel 2",
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--inputs2", action = "store", dest = "inputs2", help = "An input Sentinel-2 data (.zip) file")
    parser.add_argument("--inputs3", action = "store", dest = "inputs3", help = "An input Sentinel-3 (.zip) file")
    parser.add_argument("--SR", action = "store_true", dest = "sr", help = "Super-Resolve Sentinel-2 bands to 10 m spatial resolution")
    args = parser.parse_args()
    inputs2 = args.inputs2
    inputs3= args.inputs3
    '''
    
    '''
    inputs2 = os.path.join('Data','TernaMag','S2A_MSIL2A_20240817T090551_N0511_R050_T34SGH_20240817T134550.zip')
    inputs3 = os.path.join('Data','TernaMag', 'S3A_SL_2_LST____20240817T081626_20240817T081926_20240818T181238_0179_116_021_2340_PS1_O_NT_004.zip')
    '''
    
    '''
    inputs2 = os.path.join('Data', 'Canteras', 'S2A_MSIL2A_20240909T105621_N0511_R094_T30SVG_20240909T154450.zip')
    inputs3 = os.path.join('Data', 'Canteras', 'S3A_SL_2_LST____20240909T100107_20240909T100407_20240910T195342_0179_116_350_2340_PS1_O_NT_004.zip')
    '''
    inputs2 = os.path.join('Data', 'Tharsis', 'S2A_MSIL2A_20240902T110621_N0511_R137_T29SPB_20240902T173346.zip')
    inputs3 = os.path.join('Data', 'Tharsis', 'S3B_SL_2_LST____20240902T104444_20240902T104744_20240903T105011_0179_097_108_2340_PS2_O_NT_004.zip')

    # Super-Resolution or Resampling to 60 m
    highResFilename = os.path.join(os.path.split(inputs2)[0], "SR_" + os.path.split(inputs2)[-1].split('.')[-2] + ".tiff")
    if os.path.exists(highResFilename):
        print("A super-resolved version of the input Sentinel-2 image exists in the path.")
    else:
        print("Performing Sentinel-2 super-resolution...")
        run_Sentinel2_SR(inputs2, highResFilename)


    ############################################################################
    # Extract Sentinel 2 scene classification mask and mask out highres file (np.nan)
    s2_mask = os.path.join(os.path.split(inputs2)[0], "MASK_" + os.path.split(inputs2)[-1].split('.')[-2] + ".tiff")
    if os.path.exists(s2_mask):
        os.remove(s2_mask)

    utils.mask_extractor(inputs2, s2_mask)
    ############################################################################
    # Unzip Sentinel 3 SLSTR
    lowResFilename = inputs3.replace(".zip", '.SEN3')

    print(lowResFilename)

    if os.path.exists(lowResFilename):
        print("Sentinel 3 SLSTR is unzipped.")
    else:
        print("Unzipping file...")
        try:
            with ZipFile(inputs3, "r") as zipObj:
                zipObj.extractall(os.path.split(inputs3)[0])
        except:
            print("Failed to unzip.")


    # Read the low-resolution dataset - project and convert to geotiff
    utils.s3_preprocessor(lowResFilename, highResFilename)

    s3_mask = os.path.join(os.path.split(lowResFilename)[0], "Subset_Flag_" + os.path.split(lowResFilename)[-1].split('.')[-2] + ".tiff")
    lowResFilename_reprojected = os.path.join(os.path.split(lowResFilename)[0], "Subset_" + os.path.split(lowResFilename)[-1].split('.')[-2] + ".tiff")

    commonOpts = {"highResFile":                    highResFilename,
                  "lowResFile":                     lowResFilename_reprojected,
                  "highResQualityFile":         s2_mask,
                  "lowResQualityFile":          s3_mask,
                  "highResGoodQualityFlags":        [4, 5, 7],      # Sentinel 2
                  "lowResGoodQualityFlags":         [0],            # Sentinel 3
                  "cvHomogeneityThreshold":         0.2,
                  "movingWindowSize":               15 * 100,
                  "disaggregatingTemperature":      True}

    dtOpts =     {"perLeafLinearRegression":    True,
                  "linearRegressionExtrapolationRatio": 0.25}

    sknnOpts =   {'hidden_layer_sizes':         (10,),
                  'activation':                 'tanh'}

    nnOpts =     {"regressionType":             REG_sklearn_ann,
                  "regressorOpt":               sknnOpts}

    useDecisionTree = True

    start_time = time.time()

    if useDecisionTree:
        opts = commonOpts.copy()
        opts.update(dtOpts)
        disaggregator = DecisionTreeSharpener(**opts)
    else:
        opts = commonOpts.copy()
        opts.update(nnOpts)
        disaggregator = NeuralNetworkSharpener(**opts)

    local_result = os.path.join(os.path.split(lowResFilename_reprojected)[0], "Local_" + os.path.split(lowResFilename_reprojected)[-1].split('.')[-2] + ".tiff")
    global_result = os.path.join(os.path.split(lowResFilename_reprojected)[0], "Global_" + os.path.split(lowResFilename_reprojected)[-1].split('.')[-2] + ".tiff")
    combined_result = os.path.join(os.path.split(lowResFilename_reprojected)[0], "Combined_" + os.path.split(lowResFilename_reprojected)[-1].split('.')[-2] + ".tiff")


    if os.path.exists(local_result) and os.path.exists(global_result):
        print('Already sharpened')
    else:
        print("Training regressor...")
        disaggregator.trainSharpener()

        print("Sharpening...")
        disaggregator.applySharpener()

    # Combination of results
    downscaledFile = disaggregator.combination(local_result, global_result)

    print("Residual analysis...")
    disaggregator.residualAnalysis(combined_result)
