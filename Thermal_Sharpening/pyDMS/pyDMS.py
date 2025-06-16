import math
import os

import numpy as np
from osgeo import gdal
from sklearn import tree, linear_model, ensemble, preprocessing
import sklearn.neural_network as ann_sklearn

from tqdm import tqdm
from tqdm.contrib import itertools
import pandas as pd

import pyDMS.pyDMSUtils as utils


REG_sknn_ann = 0
REG_sklearn_ann = 1


class DecisionTreeRegressorWithLinearLeafRegression(tree.DecisionTreeRegressor):
    ''' Decision tree regressor with added linear (bayesian ridge) regression
    for all the data points falling within each decision tree leaf node.
    Parameters
    ----------
    linearRegressionExtrapolationRatio: float (optional, default: 0.25)
        A limit on extrapolation allowed in the per-leaf linear regressions.
        The ratio is multiplied by the range of values present in each leaves'
        training dataset and added (substracted) to the maxiumum (minimum)
        value.
    decisionTreeRegressorOpt: dictionary (optional, default: {})
        Options to pass to DecisionTreeRegressor constructor. See
        http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
        for possibilities.
    Returns
    -------
    None
    '''
    def __init__(self, linearRegressionExtrapolationRatio=0.25, decisionTreeRegressorOpt={}):
        super(DecisionTreeRegressorWithLinearLeafRegression, self).__init__(**decisionTreeRegressorOpt)
        self.decisionTreeRegressorOpt = decisionTreeRegressorOpt
        self.leafParameters = {}
        self.linearRegressionExtrapolationRatio = linearRegressionExtrapolationRatio

    def fit(self, X, y, sample_weight, fitOpt={}):
        ''' Build a decision tree regressor from the training set (X, y).
        Parameters
        ----------
        X: array-like or sparse matrix, shape = [n_samples, n_features]
            The training input samples. Internally, it will be converted to
            dtype=np.float32 and if a sparse matrix is provided to a sparse
            csc_matrix.
        y: array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (real numbers). Use dtype=np.float64 and
            order='C' for maximum efficiency.
        sample_weight: array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node.
        fitOpt: dictionary (optional, default: {})
            Options to pass to DecisionTreeRegressor fit function. See
            http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
            for possibilities.
        Returns
        -------
        Self
        '''

        # Fit a normal regression tree
        super(DecisionTreeRegressorWithLinearLeafRegression, self).fit(X, y, sample_weight,
                                                                       **fitOpt)

        # Create a linear regression for all input points which fall into
        # one output leaf
        predictedValues = super(DecisionTreeRegressorWithLinearLeafRegression, self).predict(X)
        leafValues = np.unique(predictedValues)
        for value in leafValues:
            ind = predictedValues == value
            leafLinearRegrsion = linear_model.BayesianRidge()
            leafLinearRegrsion.fit(X[ind, :], y[ind])
            self.leafParameters[value] = {"linearRegression": leafLinearRegrsion,
                                          "max": np.max(y[ind]),
                                          "min": np.min(y[ind])}

        return self

    def predict(self, X, predictOpt={}):
        ''' Predict class or regression value for X.
        Parameters
        ----------
        X: array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            dtype=np.float32 and if a sparse matrix is provided to a sparse
            csr_matrix.
        predictOpt: dictionary (optional, default: {})
            Options to pass to DecisionTreeRegressor predict function. See
            http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
            for possibilities.
        Returns
        -------
        y: array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted classes, or the predict values.
        '''

        # Do normal regression tree prediction
        y = super(DecisionTreeRegressorWithLinearLeafRegression, self).predict(X, **predictOpt)

        # And also apply per-leaf linear regression
        for leafValue in self.leafParameters.keys():
            ind = y == leafValue
            if X[ind, :].size > 0:
                y[ind] = self.leafParameters[leafValue]["linearRegression"].predict(X[ind, :])
                # Limit extrapolation
                extrapolationRange = self.linearRegressionExtrapolationRatio * (
                                        self.leafParameters[leafValue]["max"] -
                                        self.leafParameters[leafValue]["min"])
                y[ind] = np.maximum(y[ind],
                                    self.leafParameters[leafValue]["min"] - extrapolationRange)
                y[ind] = np.minimum(y[ind],
                                    self.leafParameters[leafValue]["max"] + extrapolationRange)

        return y


class DecisionTreeSharpener(object):
    ''' Decision tree based sharpening (disaggregation) of low-resolution
    images using high-resolution images. The implementation is mostly based on [Gao2012].
    Decision tree based regressor is trained with high-resolution data resampled to
    low resolution and low-resolution data and then applied
    directly to high-resolution data to obtain high-resolution representation
    of the low-resolution data.
    The implementation includes selecting training data based on homogeneity
    statistics and using the homogeneity as weight factor ([Gao2012], section 2.2),
    performing linear regression with samples located within each regression
    tree leaf node ([Gao2012], section 2.1), using an ensemble of regression trees
    ([Gao2012], section 2.1), performing local (moving window) and global regression and
    combining them based on residuals ([Gao2012] section 2.3) and performing residual
    analysis and bias correction ([Gao2012], section 2.4)
    Parameters
    ----------
    highResFile: list of strings
        A list of file paths to high-resolution images to be used during the
        training of the sharpener.
    lowResFile: list of strings
        A list of file paths to low-resolution images to be used during the
        training of the sharpener. There must be one low-resolution image
        for each high-resolution image.
    lowResQualityFile: list of strings (optional, default: [])
        A list of file paths to low-resolution quality images to be used to
        mask out low-quality low-resolution pixels during training. If provided
        there must be one quality image for each low-resolution image.
    lowResGoodQualityFlags: list of integers (optional, default: [])
        A list of values indicating which pixel values in the low-resolution
        quality images should be considered as good quality.
    cvHomogeneityThreshold: float (optional, default: 0)
        A threshold of coeficient of variation below which high-resolution
        pixels resampled to low-resolution are considered homogeneous and
        usable during the training of the disaggregator. If threshold is 0 or
        negative then it is set automatically such that 80% of pixels are below
        it.
    movingWindowSize: integer (optional, default: 0)
        The size of local regression moving window in low-resolution pixels. If
        set to 0 then only global regression is performed.
    disaggregatingTemperature: boolean (optional, default: False)
        Flag indicating whether the parameter to be disaggregated is
        temperature (e.g. land surface temperature). If that is the case then
        at some points it needs to be converted into radiance. This is becasue
        sensors measure energy, not temperature, plus radiance is the physical
        measurements it makes sense to average, while radiometric temperature
        behaviour is not linear.
    perLeafLinearRegression: boolean (optional, default: True)
        Flag indicating if linear regression should be performed on all data
        points falling within each regression tree leaf node.
    linearRegressionExtrapolationRatio: float (optional, default: 0.25)
        A limit on extrapolation allowed in the per-leaf linear regressions.
        The ratio is multiplied by the range of values present in each leaves'
        training dataset and added (substracted) to the maxiumum (minimum)
        value.
    regressorOpt: dictionary (optional, default: {})
        Options to pass to DecisionTreeRegressor constructor See
        http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
        for possibilities. Note that max_leaf_nodes and min_samples_leaf
        parameters will beoverwritten in the code.
    baggingRegressorOpt: dictionary (optional, default: {})
        Options to pass to BaggingRegressor constructor. See
        http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html
        for possibilities.
    Returns
    -------
    None
    References
    ----------
    .. [Gao2012] Gao, F., Kustas, W. P., & Anderson, M. C. (2012). A Data
       Mining Approach for Sharpening Thermal Satellite Imagery over Land.
       Remote Sensing, 4(11), 3287–3319. https://doi.org/10.3390/rs4113287
    '''
    def __init__(self,
                 highResFile,
                 lowResFile,
                 highResQualityFile,
                 lowResQualityFile,
                 lowResGoodQualityFlags=[],
                 highResGoodQualityFlags=[],
                 cvHomogeneityThreshold=0,
                 movingWindowSize=0,
                 disaggregatingTemperature=False,
                 perLeafLinearRegression=True,
                 linearRegressionExtrapolationRatio=0.25,
                 regressorOpt={},
                 baggingRegressorOpt={}):

        self.highResFile = highResFile
        self.lowResFile = lowResFile
        self.lowResQualityFile = lowResQualityFile
        self.highResQualityFile = highResQualityFile
        self.lowResGoodQualityFlags = lowResGoodQualityFlags
        self.highResGoodQualityFlags = highResGoodQualityFlags


        self.cvHomogeneityThreshold = cvHomogeneityThreshold
        # If threshold is 0 or negative then it is set automatically such that
        # 80% of pixels are below it.
        if self.cvHomogeneityThreshold <= 0:
            self.autoAdjustCvThreshold = True
            self.precentileThreshold = 80
        else:
            self.autoAdjustCvThreshold = False

        # Moving window size in low resolution pixels
        self.movingWindowSize = float(movingWindowSize)
        # The extension (on each side) by which sampling window size is larger
        # then prediction window size (see section 2.3 of Gao paper)
        self.movingWindowExtension = self.movingWindowSize * 0.25
        self.windowExtents = []

        self.disaggregatingTemperature = disaggregatingTemperature

        # Flag to determine whether a multivariate linear regression should be
        # constructed for samples in each leaf of the regression tree
        # (see section 2.1 of Gao paper)
        self.perLeafLinearRegression = perLeafLinearRegression
        self.linearRegressionExtrapolationRatio = linearRegressionExtrapolationRatio

        self.regressorOpt = regressorOpt
        self.baggingRegressorOpt = baggingRegressorOpt
        self.df = pd.DataFrame()


    def trainSharpener(self):
        ''' Train the sharpener using high- and low-resolution input files
        and settings specified in the constructor. Local (moving window) and
        global regression decision trees are trained with high-resolution data
        resampled to low resolution and low-resolution data. The training
        dataset is selected based on homogeneity of resampled high-resolution
        data being below specified threshold and quality mask (if given) of
        low resolution data. The homogeneity statistics are also used as weight
        factors for the training samples (more homogenous - higher weight).
        Parameters
        ----------
        None
        Returns
        -------
        None
        '''

        # Select good data (training samples) from low- and high-resolution
        # input images.

        scene_HR = gdal.Open(self.highResFile)
        mask_HR = gdal.Open(self.highResQualityFile)
        sizeX = scene_HR.RasterXSize
        sizeY = scene_HR.RasterYSize
        for b in range(0, scene_HR.RasterCount):
            desc = scene_HR.GetRasterBand(b+1).GetDescription()
            if desc == 'B2 (490 nm)':
                b2 = scene_HR.GetRasterBand(b+1)

            elif desc == 'B3 (560 nm)':
                b3 = scene_HR.GetRasterBand(b+1)

            elif desc == 'B4 (665 nm)':
                b4 = scene_HR.GetRasterBand(b+1)

            elif desc == 'B8 (842 nm)':
                b8 = scene_HR.GetRasterBand(b+1)

            elif desc == 'B5 (705 nm)':
                b5 = scene_HR.GetRasterBand(b+1)

            elif desc == 'B6 (740 nm)':
                b6 = scene_HR.GetRasterBand(b+1)

            elif desc == 'B7 (783 nm)':
                b7 = scene_HR.GetRasterBand(b+1)

            elif desc == 'B8A (865 nm)':
                b8A = scene_HR.GetRasterBand(b+1)

            elif desc == 'B11 (1610 nm)':
                b11 = scene_HR.GetRasterBand(b+1)

            elif desc == 'B12 (2190 nm)':
                b12 = scene_HR.GetRasterBand(b+1)

            elif desc == 'B1 (443 nm)':
                b1 = scene_HR.GetRasterBand(b+1)

            elif desc == 'B9 (945 nm)':
                b9 = scene_HR.GetRasterBand(b+1)
            else:
                print('Invalid band description')

        scene_LR = gdal.Open(self.lowResFile)
        mask_LR = gdal.Open(self.lowResQualityFile)
        lowResData = scene_LR.ReadAsArray()
        lowResQualityData = mask_LR.ReadAsArray()

        print('Extracting samples...')
        counter = 0
        self.df = pd.DataFrame(columns = ['temperature', 's2-col', 's2-row', 'weight', 'features'])
        for i, j in itertools.product(range(lowResData.shape[0]), range(lowResData.shape[1])):
            if lowResData[i,j] != -32768.0 and lowResQualityData[i,j] in self.lowResGoodQualityFlags:

                xsize = 100 if j * 100 + 100 < sizeX else sizeX - j * 100
                ysize = 100 if i * 100 + 100 < sizeY else sizeY - i * 100

                mask_data = mask_HR.ReadAsArray(j * 100, i * 100, xsize, ysize)
                t1 = b1.ReadAsArray(j * 100, i * 100, xsize, ysize).astype(float)
                t2 = b2.ReadAsArray(j * 100, i * 100, xsize, ysize).astype(float)
                t3 = b3.ReadAsArray(j * 100, i * 100, xsize, ysize).astype(float)
                t4 = b4.ReadAsArray(j * 100, i * 100, xsize, ysize).astype(float)
                t5 = b5.ReadAsArray(j * 100, i * 100, xsize, ysize).astype(float)
                t6 = b6.ReadAsArray(j * 100, i * 100, xsize, ysize).astype(float)
                t7 = b7.ReadAsArray(j * 100, i * 100, xsize, ysize).astype(float)
                t8 = b8.ReadAsArray(j * 100, i * 100, xsize, ysize).astype(float)
                t8a = b8A.ReadAsArray(j * 100, i * 100, xsize, ysize).astype(float)
                t9 = b9.ReadAsArray(j * 100, i * 100, xsize, ysize).astype(float)
                t11 = b11.ReadAsArray(j * 100, i * 100, xsize, ysize).astype(float)
                t12 = b12.ReadAsArray(j * 100, i * 100, xsize, ysize).astype(float)

                sum = 0
                for a in range(mask_data.shape[0]):
                    for b in range(mask_data.shape[1]):
                        if mask_data[a, b] in self.highResGoodQualityFlags:
                            sum += 1
                        else:
                            t1[a, b] = t2[a, b] =  t3[a, b] =  t4[a, b] =  t5[a, b] =  t6[a, b] =  t7[a, b] =  t8[a, b] =  t8a[a, b] = t9[a, b] = t11[a, b] = t12[a,b] = np.nan

                if sum > 0:

                    z = np.divide(np.nanstd(t1), np.nanmean(t1)) + np.divide(np.nanstd(t2), np.nanmean(t2)) + \
                    np.divide(np.nanstd(t3), np.nanmean(t3)) + np.divide(np.nanstd(t4), np.nanmean(t4)) + \
                    np.divide(np.nanstd(t5), np.nanmean(t5)) + np.divide(np.nanstd(t6), np.nanmean(t6)) + \
                    np.divide(np.nanstd(t7), np.nanmean(t7)) + np.divide(np.nanstd(t8), np.nanmean(t8)) + \
                    np.divide(np.nanstd(t8a), np.nanmean(t8a)) + np.divide(np.nanstd(t9), np.nanmean(t9)) + \
                    np.divide(np.nanstd(t11), np.nanmean(t11)) + np.divide(np.nanstd(t12), np.nanmean(t12))

                    cv = z/12

                    if cv < self.cvHomogeneityThreshold:
                        self.df = pd.concat([self.df, pd.DataFrame.from_records([{'temperature': lowResData[i,j], 's2-col':j*100, 's2-row':i*100, 'weight':cv, 'features': np.array([np.nanmean(t1), np.nanmean(t2), np.nanmean(t3), np.nanmean(t4), np.nanmean(t5), np.nanmean(t6), np.nanmean(t7), np.nanmean(t8), np.nanmean(t8a), np.nanmean(t9), np.nanmean(t11), np.nanmean(t12)])}])])

                counter += 1


        csv_data = self.df.to_csv(os.path.join(os.path.split(self.highResFile)[0], "samples.csv"), columns=['temperature', 's2-col', 's2-row', 'weight', 'features'], index=False)

        print('Number of available samples:', counter)
        print('Number of homogeneous samples actually used:', self.df.shape)

        scene_HR = None
        mask_HR = None
        scene_LR = None
        mask_LR = None

        windows = []
        extents = []

        for y in range(int(math.ceil(sizeY/self.movingWindowSize))):
            for x in range(int(math.ceil(sizeX/self.movingWindowSize))):
                windows.append([int(max(y * self.movingWindowSize - self.movingWindowExtension, 0)),
                                int(min((y+1) * self.movingWindowSize + self.movingWindowExtension, sizeY)),
                                int(max(x * self.movingWindowSize - self.movingWindowExtension, 0)),
                                int(min((x+1) * self.movingWindowSize + self.movingWindowExtension, sizeX))])
                extents.append([int(max(y * self.movingWindowSize, 0)),
                                int(min((y+1) * self.movingWindowSize, sizeY)),
                                int(max(x * self.movingWindowSize, 0)),
                                int(min((x+1) * self.movingWindowSize, sizeX))])

        windows.append([0, sizeY, 0, sizeX]) # add global
        self.windowExtents = extents

        print('Training...')
        windowsNum = len(windows)
        self.reg = [None for _ in range(windowsNum)]
        for i in tqdm(range(windowsNum)):
            if i < windowsNum-1:
                local = True
            else:
                local = False
            select_df = self.df.loc[(self.df['s2-row'] >= windows[i][0]) & (self.df['s2-row'] < windows[i][1]) & \
                (self.df['s2-col'] >=  windows[i][2]) & (self.df['s2-col'] < windows[i][3])]
            if select_df.shape[0] > 0:
                input = np.vstack(select_df['features'])
                output = select_df['temperature']
                weight = select_df['weight']
                if input.shape[0] > 0:
                    self.reg[i] = self._doFit(output, input, weight, local)


    def applySharpener(self):
        ''' Apply the trained sharpener to a given high-resolution image to
        derive corresponding disaggregated low-resolution image. If local
        regressions were used during training then they will only be applied
        where their moving window extent overlaps with the high resolution
        image passed to this function. Global regression will be applied to the
        whole high-resolution image wihtout geographic constraints.
        Parameters
        ----------
        highResFilename: string
            Path to the high-resolution image file do be used during
            disaggregation.
        lowResFilename: string (optional, default: None)
            Path to the low-resolution image file corresponding to the
            high-resolution input file. If local regressions
            were trained and low-resolution filename is given then the local
            and global regressions will be combined based on residual values of
            the different regressions to the low-resolution image (see [Gao2012]
            2.3). If local regressions were trained and low-resolution
            filename is not given then only the local regressions will be used.
        Returns
        -------
        outImage: GDAL memory file object
            The file object contains an in-memory, georeferenced disaggregator
            output.
        '''

        # Open and read the high resolution input file
        scene_HR = gdal.Open(self.highResFile)
        for b in range(0, scene_HR.RasterCount):
            desc = scene_HR.GetRasterBand(b+1).GetDescription()
            if desc == 'B2 (490 nm)':
                b2 = scene_HR.GetRasterBand(b+1)

            elif desc == 'B3 (560 nm)':
                b3 = scene_HR.GetRasterBand(b+1)

            elif desc == 'B4 (665 nm)':
                b4 = scene_HR.GetRasterBand(b+1)

            elif desc == 'B8 (842 nm)':
                b8 = scene_HR.GetRasterBand(b+1)

            elif desc == 'B5 (705 nm)':
                b5 = scene_HR.GetRasterBand(b+1)

            elif desc == 'B6 (740 nm)':
                b6 = scene_HR.GetRasterBand(b+1)

            elif desc == 'B7 (783 nm)':
                b7 = scene_HR.GetRasterBand(b+1)

            elif desc == 'B8A (865 nm)':
                b8A = scene_HR.GetRasterBand(b+1)

            elif desc == 'B11 (1610 nm)':
                b11 = scene_HR.GetRasterBand(b+1)

            elif desc == 'B12 (2190 nm)':
                b12 = scene_HR.GetRasterBand(b+1)

            elif desc == 'B1 (443 nm)':
                b1 = scene_HR.GetRasterBand(b+1)

            elif desc == 'B9 (945 nm)':
                b9 = scene_HR.GetRasterBand(b+1)
            else:
                print('Invalid band description')

        mask_HR = gdal.Open(self.highResQualityFile)
        sizeX = scene_HR.RasterXSize
        sizeY = scene_HR.RasterYSize

        driver = gdal.GetDriverByName( 'GTiff' )
        out_local = driver.Create(os.path.join(os.path.split(self.lowResFile)[0], "Local_" + os.path.split(self.lowResFile)[-1].split('.')[-2] + ".tiff"), sizeX,  sizeY, 1, gdal.GDT_Float32)
        out_global = driver.Create(os.path.join(os.path.split(self.lowResFile)[0], "Global_" + os.path.split(self.lowResFile)[-1].split('.')[-2] + ".tiff"), sizeX,  sizeY, 1, gdal.GDT_Float32)

        out_local.SetGeoTransform(scene_HR.GetGeoTransform())
        out_local.SetProjection(scene_HR.GetProjection())
        out_local.GetRasterBand(1).SetNoDataValue(np.nan)

        out_global.SetGeoTransform(scene_HR.GetGeoTransform())
        out_global.SetProjection(scene_HR.GetProjection())
        out_global.GetRasterBand(1).SetNoDataValue(np.nan)


        # Do the downscailing on the moving windows if there are any
        for i, extent in enumerate(tqdm(self.windowExtents)):

            mask_data = mask_HR.ReadAsArray(extent[2], extent[0], extent[3]-extent[2], extent[1]-extent[0])

            t1 = b1.ReadAsArray(extent[2], extent[0], extent[3]-extent[2], extent[1]-extent[0]).astype(float)
            t2 = b2.ReadAsArray(extent[2], extent[0], extent[3]-extent[2], extent[1]-extent[0]).astype(float)
            t3 = b3.ReadAsArray(extent[2], extent[0], extent[3]-extent[2], extent[1]-extent[0]).astype(float)
            t4 = b4.ReadAsArray(extent[2], extent[0], extent[3]-extent[2], extent[1]-extent[0]).astype(float)
            t5 = b5.ReadAsArray(extent[2], extent[0], extent[3]-extent[2], extent[1]-extent[0]).astype(float)
            t6 = b6.ReadAsArray(extent[2], extent[0], extent[3]-extent[2], extent[1]-extent[0]).astype(float)
            t7 = b7.ReadAsArray(extent[2], extent[0], extent[3]-extent[2], extent[1]-extent[0]).astype(float)
            t8 = b8.ReadAsArray(extent[2], extent[0], extent[3]-extent[2], extent[1]-extent[0]).astype(float)
            t8a = b8A.ReadAsArray(extent[2], extent[0], extent[3]-extent[2], extent[1]-extent[0]).astype(float)
            t9 = b9.ReadAsArray(extent[2], extent[0], extent[3]-extent[2], extent[1]-extent[0]).astype(float)
            t11 = b11.ReadAsArray(extent[2], extent[0], extent[3]-extent[2], extent[1]-extent[0]).astype(float)
            t12 = b12.ReadAsArray(extent[2], extent[0], extent[3]-extent[2], extent[1]-extent[0]).astype(float)

            windowsInData = np.stack([t1, t2, t3, t4, t5, t6, t7, t8, t8a, t9, t11, t12], axis = -1)

            # Local
            if self.reg[i] is not None:
                outWindowData = self._doPredict(windowsInData, self.reg[i])

            # Global
            if self.reg[-1] is not None:
                outFullData = self._doPredict(windowsInData, self.reg[-1])

            outWindowData_Masked = np.array([[outWindowData[k,j] if mask_data[k,j] in self.highResGoodQualityFlags else np.nan for j in range(mask_data.shape[1])] for k in range(mask_data.shape[0])])
            outFullData_Masked = np.array([[outFullData[k,j] if mask_data[k,j] in self.highResGoodQualityFlags else np.nan for j in range(mask_data.shape[1])] for k in range(mask_data.shape[0])])

            out_local.GetRasterBand(1).WriteArray(outWindowData_Masked, xoff = extent[2], yoff = extent[0])
            out_local.FlushCache()

            out_global.GetRasterBand(1).WriteArray(outFullData_Masked, xoff = extent[2], yoff = extent[0])
            out_global.FlushCache()


        scene_HR = None
        mask_HR = None


        '''
        # Combine the windowed and whole image regressions
        # If there is no windowed regression just use the whole image regression
        if np.all(np.isnan(outWindowData)):
            outData = outFullData
        # If corresponding low resolution file is provided then combine the two
        # regressions based on residuals (see section 2.3 of Gao paper)
        elif lowResFilename is not None:
            lowResScene = gdal.Open(lowResFilename)
            outWindowScene = utils.saveImg(outWindowData,
                                           highResFile.GetGeoTransform(),
                                           highResFile.GetProjection(),
                                           "MEM",
                                           noDataValue=np.nan)
            windowedResidual, _, _ = self._calculateResidual(outWindowScene, lowResScene)
            outWindowScene = None
            outFullScene = utils.saveImg(outFullData,
                                         highResFile.GetGeoTransform(),
                                         highResFile.GetProjection(),
                                         "MEM",
                                         noDataValue=np.nan)
            fullResidual, _, _ = self._calculateResidual(outFullScene, lowResScene)
            outFullScene = None
            lowResScene = None
            # windowed weight
            ww = (1/windowedResidual)**2/((1/windowedResidual)**2 + (1/fullResidual)**2)
            # full weight
            fw = 1 - ww
            outData = outWindowData*ww + outFullData*fw
        # Otherwised use just windowed regression
        else:
            outData = outWindowData

        # Fix NaN's
        nanInd = np.any(nanInd, -1)
        outData[nanInd] = np.nan

        outImage = utils.saveImg(outData,
                                 highResFile.GetGeoTransform(),
                                 highResFile.GetProjection(),
                                 "MEM",
                                 noDataValue=np.nan)

        highResFile = None
        inData = None
        return outImage
        '''

    def combination(self, local_result, global_result):

        scene_LR = gdal.Open(self.lowResFile)
        mask_LR = gdal.Open(self.lowResQualityFile)
        lowResData = scene_LR.ReadAsArray()
        lowResQualityData = mask_LR.ReadAsArray()

        windowedResidual, _ = self._calculateResidual(local_result, os.path.join(os.path.split(self.lowResFile)[0], "Local_Res_" + os.path.split(self.lowResFile)[-1].split('.')[-2] + ".tiff"))
        globalResidual, _ = self._calculateResidual(global_result, os.path.join(os.path.split(self.lowResFile)[0], "Global_Res_" + os.path.split(self.lowResFile)[-1].split('.')[-2] + ".tiff"))

        ww = (1/windowedResidual)**2/((1/windowedResidual)**2 + (1/globalResidual)**2)
        # full weight
        fw = 1 - ww

        local = gdal.Open(local_result)
        global_ = gdal.Open(global_result)

        local_values = local.GetRasterBand(1).ReadAsArray()
        global_values = global_.GetRasterBand(1).ReadAsArray()
        combinedData = local_values*ww + global_values*fw

        driver = gdal.GetDriverByName( 'GTiff' )
        out_combined = driver.Create(os.path.join(os.path.split(self.lowResFile)[0], "Combined_" + os.path.split(self.lowResFile)[-1].split('.')[-2] + ".tiff"), combinedData.shape[1],  combinedData.shape[0], 1, gdal.GDT_Float32)

        out_combined.SetGeoTransform(local.GetGeoTransform())
        out_combined.SetProjection(local.GetProjection())
        out_combined.GetRasterBand(1).SetNoDataValue(np.nan)

        out_combined.GetRasterBand(1).WriteArray(combinedData)
        out_combined.FlushCache()

        return combinedData


    def _calculateResidual(self, result, outname):

        scene_LR = gdal.Open(self.lowResFile)
        mask_LR = gdal.Open(self.lowResQualityFile)
        lowResData = scene_LR.ReadAsArray()
        lowResQualityData = mask_LR.ReadAsArray()

        result_ = gdal.Open(result)
        sizeX = result_.RasterXSize
        sizeY = result_.RasterYSize

        residual_= np.zeros(lowResData.shape)

        for i, j in itertools.product(range(lowResData.shape[0]), range(lowResData.shape[1])):
            if lowResQualityData[i,j] in self.lowResGoodQualityFlags and lowResData[i,j] != -32768.0:

                xsize = 100 if j * 100 + 100 < sizeX else sizeX - j * 100
                ysize = 100 if i * 100 + 100 < sizeY else sizeY - i * 100

                result_values = result_.GetRasterBand(1).ReadAsArray(j * 100, i * 100, xsize, ysize).astype(float)

                agg_temp = np.nan if np.all(result_values!=result_values) else np.nanmean(result_values**4)

                residual_[i,j] = lowResData[i,j]**4 - agg_temp

        residual_smoothed = utils.binomialSmoother(residual_)

        driver = gdal.GetDriverByName("MEM")
        out_file = driver.Create("MEM", lowResData.shape[1],  lowResData.shape[0], 1, gdal.GDT_Float32)

        out_file.SetGeoTransform(scene_LR.GetGeoTransform())
        out_file.SetProjection(scene_LR.GetProjection())
        out_file.GetRasterBand(1).SetNoDataValue(np.nan)
        out_file.GetRasterBand(1).WriteArray(residual_smoothed)
        out_file.FlushCache()

        minx = result_.GetGeoTransform()[0]
        maxy = result_.GetGeoTransform()[3]
        maxx = minx + result_.GetGeoTransform()[1] * result_.RasterXSize
        miny = maxy + result_.GetGeoTransform()[5] * result_.RasterYSize

        outDs = gdal.Warp("",
                        out_file,
                        format="MEM",
                        dstSRS=result_.GetProjection(),
                        xRes=result_.GetGeoTransform()[1],
                        yRes=result_.GetGeoTransform()[5],
                        outputBounds=(minx, miny, maxx, maxy),
                        resampleAlg="bilinear")

        residual_hr = outDs.GetRasterBand(1).ReadAsArray()

        for i in range(1, residual_hr.shape[0] - 1):
            for j in range(1, residual_hr.shape[1] - 1):
                if np.isnan(residual_hr[i, j]):
                    residual_hr[i, j] = utils.removeEdgeNaNs(residual_hr, i, j)


        return residual_hr, residual_



    def residualAnalysis(self, disaggregatedFile):
        ''' Perform residual analysis and (optional) correction on the
        disaggregated file (see [Gao2012] 2.4).
        Parameters
        ----------
        disaggregatedFile: string or GDAL file object
            If string, path to the disaggregated image file; if gdal file
            object, the disaggregated image.
        lowResFilename: string
            Path to the low-resolution image file corresponding to the
            high-resolution disaggregated image.
        lowResQualityFilename: string (optional, default: None)
            Path to low-resolution quality image file. If provided then low
            quality values are masked out during residual analysis. Otherwise
            all values are considered to be of good quality.
        doCorrection: boolean (optional, default: True)
            Flag indication whether residual (bias) correction should be
            performed or not.
        Returns
        -------
        residualImage: GDAL memory file object
            The file object contains an in-memory, georeferenced residual image.
        correctedImage: GDAL memory file object
            The file object contains an in-memory, georeferenced residual
            corrected disaggregated image, or None if doCorrection was set to
            False.
        '''


        residual_HR, residual_LR = self._calculateResidual(disaggregatedFile, os.path.join(os.path.split(self.lowResFile)[0], "Combined_Res_" + os.path.split(self.lowResFile)[-1].split('.')[-2] + ".tiff"))

        scene_HR = gdal.Open(disaggregatedFile)
        scene_LR = gdal.Open(self.lowResFile)

        corrected = (residual_HR + scene_HR.GetRasterBand(1).ReadAsArray()**4)**0.25

        driver = gdal.GetDriverByName( 'GTiff' )
        out_file = driver.Create( os.path.join(os.path.split(self.lowResFile)[0], "Corrected_" + os.path.split(self.lowResFile)[-1].split('.')[-2] + ".tiff"), scene_HR.RasterXSize,  scene_HR.RasterYSize, 1, gdal.GDT_Float32)

        out_file.SetGeoTransform(scene_HR.GetGeoTransform())
        out_file.SetProjection(scene_HR.GetProjection())
        out_file.GetRasterBand(1).SetNoDataValue(np.nan)
        out_file.GetRasterBand(1).WriteArray(corrected)
        out_file.FlushCache()

        # Convert residual back to temperature for easier visualisation
        residual_LR = (residual_LR + 273.15**4)**0.25 - 273.15

        out_file = driver.Create(os.path.join(os.path.split(self.lowResFile)[0], "Corrected_Res_LR_" + os.path.split(self.lowResFile)[-1].split('.')[-2] + ".tiff"), scene_LR.RasterXSize,  scene_LR.RasterYSize, 1, gdal.GDT_Float32)

        out_file.SetGeoTransform(scene_LR.GetGeoTransform())
        out_file.SetProjection(scene_LR.GetProjection())
        out_file.GetRasterBand(1).SetNoDataValue(np.nan)
        out_file.GetRasterBand(1).WriteArray(residual_LR)
        out_file.FlushCache()

        print("LR residual bias: "+str(np.nanmean(residual_LR)))
        print("LR residual RMSD: "+str(np.nanmean(residual_LR**2)**0.5))

        scene_HR = None
        scene_LR = None


    def _doFit(self, goodData_LR, goodData_HR, weight, local):
        ''' Private function. Fits the regression tree.
        '''

        # For local regression constrain the number of tree
        # nodes (rules) - section 2.3
        if local:
            self.regressorOpt["max_leaf_nodes"] = 10
        else:
            self.regressorOpt["max_leaf_nodes"] = 30
        self.regressorOpt["min_samples_leaf"] = 10

        # If per leaf linear regression is used then use modified
        # DecisionTreeRegressor. Otherwise use the standard one.
        if self.perLeafLinearRegression:
            baseRegressor = \
                DecisionTreeRegressorWithLinearLeafRegression(self.linearRegressionExtrapolationRatio,
                                                              self.regressorOpt)
        else:
            baseRegressor = \
                tree.DecisionTreeRegressor(**self.regressorOpt)

        reg = ensemble.BaggingRegressor(baseRegressor, **self.baggingRegressorOpt)
        if goodData_HR.shape[0] <= 1:
            reg.max_samples = 1.0
        reg = reg.fit(goodData_HR, goodData_LR, sample_weight=weight)

        return reg

    def _doPredict(self, inData, reg):
        ''' Private function. Calls the regression tree.
        '''

        origShape = inData.shape
        if len(origShape) == 3:
            bands = origShape[2]
        else:
            bands = 1
        # Do the actual decision tree regression
        inData = inData.reshape((-1, bands))
        outData = reg.predict(inData)
        outData = outData.reshape((origShape[0], origShape[1]))

        return outData




class NeuralNetworkSharpener(DecisionTreeSharpener):
    ''' Neural Network based sharpening (disaggregation) of low-resolution
    images using high-resolution images. The implementation is mostly based on [Gao2012] as
    implemented in DescisionTreeSharpener except that Decision Tree regressor is replaced by
    Neural Network regressor.
    Nerual network based regressor is trained with high-resolution data resampled to
    low resolution and low-resolution data and then applied
    directly to high-resolution data to obtain high-resolution representation
    of the low-resolution data.
    The implementation includes selecting training data based on homogeneity
    statistics and using the homogeneity as weight factor ([Gao2012], section 2.2),
    performing linear regression with samples located within each regression
    tree leaf node ([Gao2012], section 2.1), using an ensemble of regression trees
    ([Gao2012], section 2.1), performing local (moving window) and global regression and
    combining them based on residuals ([Gao2012] section 2.3) and performing residual
    analysis and bias correction ([Gao2012], section 2.4)
    Parameters
    ----------
    highResFiles: list of strings
        A list of file paths to high-resolution images to be used during the
        training of the sharpener.
    lowResFiles: list of strings
        A list of file paths to low-resolution images to be used during the
        training of the sharpener. There must be one low-resolution image
        for each high-resolution image.
    lowResQualityFiles: list of strings (optional, default: [])
        A list of file paths to low-resolution quality images to be used to
        mask out low-quality low-resolution pixels during training. If provided
        there must be one quality image for each low-resolution image.
    lowResGoodQualityFlags: list of integers (optional, default: [])
        A list of values indicating which pixel values in the low-resolution
        quality images should be considered as good quality.
    cvHomogeneityThreshold: float (optional, default: 0.25)
        A threshold of coeficient of variation below which high-resolution
        pixels resampled to low-resolution are considered homogeneous and
        usable during the training of the disaggregator.
    movingWindowSize: integer (optional, default: 0)
        The size of local regression moving window in low-resolution pixels. If
        set to 0 then only global regression is performed.
    disaggregatingTemperature: boolean (optional, default: False)
        Flag indicating whether the parameter to be disaggregated is
        temperature (e.g. land surface temperature). If that is the case then
        at some points it needs to be converted into radiance. This is becasue
        sensors measure energy, not temperature, plus radiance is the physical
        measurements it makes sense to average, while radiometric temperature
        behaviour is not linear.
    regressionType: int (optional, default: 0)
        Flag indicating whether scikit-neuralnetwork (flag value = REG_sknn_ann = 0)
        or scikit-learn (flag value = REG_sklearn_ann = 1) implementations of
        nearual network should be used. See
        https://github.com/aigamedev/scikit-neuralnetwork and
        http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
        for details.
    regressorOpt: dictionary (optional, default: {})
        Options to pass to neural network regressor constructor See links in
        regressionType parameter description for details.
    baggingRegressorOpt: dictionary (optional, default: {})
        Options to pass to BaggingRegressor constructor. See
        http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html
        for possibilities.
    Returns
    -------
    None
    References
    ----------
    .. [Gao2012] Gao, F., Kustas, W. P., & Anderson, M. C. (2012). A Data
       Mining Approach for Sharpening Thermal Satellite Imagery over Land.
       Remote Sensing, 4(11), 3287–3319. https://doi.org/10.3390/rs4113287
    '''

    def __init__(self,
                 highResFile,
                 lowResFile,
                 highResQualityFile,
                 lowResQualityFile,
                 lowResGoodQualityFlags=[],
                 highResGoodQualityFlags=[],
                 cvHomogeneityThreshold=0.25,
                 movingWindowSize=0,
                 disaggregatingTemperature=False,
                 regressionType=REG_sknn_ann,
                 regressorOpt={},
                 baggingRegressorOpt={}):

        super(NeuralNetworkSharpener, self).__init__(highResFile,
                                                     lowResFile,
                                                     highResQualityFile,
                                                     lowResQualityFile,
                                                     lowResGoodQualityFlags,
                                                     highResGoodQualityFlags,
                                                     cvHomogeneityThreshold,
                                                     movingWindowSize,
                                                     disaggregatingTemperature,
                                                     regressorOpt,
                                                     baggingRegressorOpt)
        self.regressionType = regressionType
        # Move the import of sknn here because this library is not easy to
        # install but this shouldn't prevent the use of other parts of pyDMS.
        if self.regressionType == REG_sknn_ann:
            import sknn.mlp as ann_sknn

    def _doFit(self, goodData_LR, goodData_HR, weight, local):
        ''' Private function. Fits the neural network.
        '''

        # Once all the samples have been picked build the regression using
        # neural network approach
        print('Fitting neural network')
        HR_scaler = preprocessing.StandardScaler()
        data_HR = HR_scaler.fit_transform(goodData_HR)
        LR_scaler = preprocessing.StandardScaler()
        data_LR = LR_scaler.fit_transform(goodData_LR.reshape(-1, 1))
        if self.regressionType == REG_sknn_ann:
            layers = []
            if 'hidden_layer_sizes' in self.regressorOpt.keys():
                for layer in self.regressorOpt['hidden_layer_sizes']:
                    layers.append(ann_sknn.Layer(self.regressorOpt['activation'], units=layer))
            else:
                layers.append(ann_sknn.Layer(self.regressorOpt['activation'], units=100))
            self.regressorOpt.pop('activation')
            self.regressorOpt.pop('hidden_layer_sizes')
            output_layer = ann_sknn.Layer('Linear', units=1)
            layers.append(output_layer)
            baseRegressor = ann_sknn.Regressor(layers, **self.regressorOpt)
        else:
            baseRegressor = ann_sklearn.MLPRegressor(**self.regressorOpt)

        # NN regressors do not support sample weights.
        weight = None

        reg = ensemble.BaggingRegressor(baseRegressor, **self.baggingRegressorOpt)
        if data_HR.shape[0] <= 1:
            reg.max_samples = 1.0
        reg = reg.fit(data_HR, np.ravel(data_LR), sample_weight=weight)

        return {"reg": reg, "HR_scaler": HR_scaler, "LR_scaler": LR_scaler}

    def _doPredict(self, inData, nn):
        ''' Private function. Calls the neural network.
        '''

        reg = nn["reg"]
        HR_scaler = nn["HR_scaler"]
        LR_scaler = nn["LR_scaler"]

        origShape = inData.shape
        if len(origShape) == 3:
            bands = origShape[2]
        else:
            bands = 1

        # Do the actual neural network regression
        inData = inData.reshape((-1, bands))
        inData = HR_scaler.transform(inData)
        outData = reg.predict(inData)
        outData = LR_scaler.inverse_transform(outData)
        outData = outData.reshape((origShape[0], origShape[1]))

        return outData
