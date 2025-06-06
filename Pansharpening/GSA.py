import os
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt

from scipy import signal
from scipy import ndimage
import PIL

def alpha_estimation(a, imageHR0):

    IHc = imageHR0.reshape((imageHR0.shape[0]*imageHR0.shape[1],1), order='F')
    ILRc = a.reshape((a.shape[0]*a.shape[1], a.shape[2]),order='F')
    alpha = np.linalg.lstsq(ILRc,IHc)
    alpha = alpha[0]

    return alpha

def PanSharpening_GSA(HS, PAN):

    """

    Performs the PanSharpening GSA

    inputs:

        * HS: np.array with the HS image
        * PAN: np.array with the PAN image

    outputs:

        * I_Fus_GSA: Pansharpened image with the PAN spatial resolution and the
        hyperspectral cube spectral resolution

    """

    HS = np.moveaxis(HS, 0, -1)
    PAN = PAN[..., np.newaxis]


    ratio1 = PAN.shape[0]//HS.shape[0]

    print(ratio1)

    ratio1 = int(ratio1)


    # Aplicamos el filtro

    r_im, c_im, b_im = HS.shape
    r_pan, c_pan, b_pan = PAN.shape

    L = 45

    BaseCoeff = ratio1*signal.firwin(L,1/ratio1)

    I1LRU = np.zeros([ratio1*r_im, ratio1*c_im, b_im])

    I1LRU[0:r_pan:ratio1, 0:c_pan:ratio1, :] = HS

    m = L//2

    Filtro = np.zeros([L,L])
    Filtro[m,:] = BaseCoeff

    print('Step 1: Filter application ...')

    for nn in range(I1LRU.shape[2]):


        t = I1LRU[:,:,nn]
        t = ndimage.filters.convolve(t.T, Filtro, mode = 'wrap')
        I1LRU[:,:,nn] = ndimage.filters.convolve(t.T, Filtro, mode = 'wrap')


    imageLR = I1LRU

    #REMOVE MEANS FROM imageLR


    imageLR0 = np.zeros([imageLR.shape[0], imageLR.shape[1], imageLR.shape[2]])

    for i in range(imageLR.shape[2]):
        imageLR0[:,:,i] = imageLR[:,:,i] - np.mean(imageLR[:,:,i])

    # REMOVE MEANS FROM imageLR_LP


    imageLR_LP0 = np.zeros([HS.shape[0], HS.shape[1], HS.shape[2]])

    for ii in range(HS.shape[2]):
        imageLR_LP0[:,:,ii] = HS[:,:,ii] -np.mean(HS[:,:,ii])


    # Synthetic intensity

    print('Step 2: Calculating synthetic intensity ...')

    imageHR = PAN[:,:,0]

    imageHR0 = imageHR - np.mean(imageHR)

    image_pil = PIL.Image.fromarray(imageHR0)
    imageHR0 = image_pil.resize((HS.shape[1], HS.shape[0]), resample = PIL.Image.BICUBIC )
    imageHR0 = np.asarray(imageHR0)

    a = np.dstack((imageLR_LP0, np.ones([HS.shape[0], HS.shape[1]])))

    alpha = alpha_estimation(a, imageHR0)
    alpha = np.reshape(alpha, (1,1,len(alpha)))

    kk2 = np.tile(alpha,(imageLR.shape[0],imageLR.shape[1], 1))
    kk = np.dstack((imageLR0, np.ones([imageLR.shape[0], imageLR.shape[1]])))
    gg2 = kk * kk2
    I = np.sum(gg2, axis=2)


    # REMOVE MEAN FROM I

    I0 = I - np.mean(I)

    # COEFFICIENTS

    print('Step 3: Obtaining coefficients ...')

    g = np.ones([1,1, imageLR.shape[2] +1])

    for i in range(imageLR.shape[2]):
        h = imageLR0[:,:,i]
        c = np.cov(I0.flatten(), h.flatten())
        g[0,0,i+1] = c[0,1]/np.var(I0.flatten())

    imageHR = imageHR - np.mean(imageHR)

    # DETAIL EXTRACTION

    delta = imageHR - I0
    deltam = np.tile(delta.T.flatten(),(imageLR.shape[2]+1));
    deltam = deltam.reshape([delta.T.flatten().shape[0], imageLR.shape[2]+1], order = 'F')


    # FUSION

    print('Step 4: Performing fusion ...')

    V = I0.T.flatten()

    for ii in range(imageLR.shape[2]):
        h = imageLR0[:,:,ii]
        V = np.concatenate((V,h.T.flatten()))

    V = V.reshape([delta.T.flatten().shape[0], imageLR.shape[2]+1], order = 'F')

    gm = np.zeros([V.shape[0], V.shape[1]])

    for ii in range(g.shape[2]):
        pp = np.squeeze(g[0,0,ii]) * np.ones([imageLR.shape[0] * imageLR.shape[1],1])
        gm[:,ii] = pp.T

    V_hat = V + deltam * gm

    # RESHAPE FUSION RESULT


    V_hat_r = V_hat[:,1:V_hat.shape[1]]

    I_Fus_GSA = V_hat_r.reshape([imageLR.shape[0], imageLR.shape[1], imageLR.shape[2]],
                               order = 'F');

    # FINAL MEAN EQUALIZATION

    print('Step 5: Reshape the fused matrix ...')

    for ii in range(imageLR.shape[2]):
        h = I_Fus_GSA[:,:,ii]
        I_Fus_GSA[:,:,ii] = h - np.mean(h) + np.mean(imageLR[:,:,ii])



    return I_Fus_GSA

if __name__ == "__main__":

    # Add the paths of the coregistered PRISMA Panchromatic and Hyperspectral images
    hs_path = os.path.join('Processed_Dataset','2_PRISMA_Datacubes','Tharsis', 'PRS_L2D_STD_20200224112618_20200224112623_0001_HS_coreg.tiff')
    pan_path = os.path.join('Processed_Dataset','2_PRISMA_Datacubes','Tharsis', 'PRS_L2D_STD_20200224112618_20200224112623_0001_PAN_coreg.tiff')

    hs_file = gdal.Open(hs_path)
    pan_file = gdal.Open(pan_path)

    
    hs_data = hs_file.ReadAsArray()
    pan_data = pan_file.ReadAsArray()

    PanSharp_np = PanSharpening_GSA(hs_data, pan_data)


    PanSharp_np = np.moveaxis(PanSharp_np, -1, 0)

    gt = pan_file.GetGeoTransform()
    band = pan_file.GetRasterBand(1)
    proj = pan_file.GetProjection()
    band_num = hs_file.RasterCount

    final_path = os.path.join('Processed_Dataset','2_PRISMA_Datacubes','Tharsis', 'PRS_L2D_STD_20200224112618_20200224112623_0001_pansharp.tiff')
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(final_path, pan_file.RasterXSize, pan_file.RasterYSize, band_num, band.DataType)


    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(proj)

    out_ds.WriteArray(PanSharp_np)

    # Loop through bands
    for i in range(1, band_num + 1):
        in_band = hs_file.GetRasterBand(i)
        out_band = out_ds.GetRasterBand(i)

        # Copy band metadata
        out_band.SetDescription(in_band.GetDescription())
        if in_band.GetNoDataValue() is not None:
            out_band.SetNoDataValue(in_band.GetNoDataValue())

    out_ds.FlushCache()
    out_ds = None
