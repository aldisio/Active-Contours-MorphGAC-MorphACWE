import morphsnakes

import numpy as np
#from scipy.misc import imread
from imageio import imread
from matplotlib import pyplot as ppl

import sys
sys.path.append('roi')
from startPoint import getInitialPoint 
import cv2

def rgb2gray(img):
    """Convert a RGB image to gray scale."""
    return 0.2989*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]

def circle_levelset(shape, center, sqradius, scalerow=1.0):
    """Build a binary function with a circle as the 0.5-levelset."""
    print("center:")
    print(center)

    grid = np.mgrid[list(map(slice, shape))].T - center
    phi = sqradius - np.sqrt(np.sum((grid.T)**2, 0))
    u = np.float_(phi > 0)
    return u

def test_nodule(img):
    # Load the image.
    #img = imread("testimages/mama07ORI.bmp")[...,0]/255.0    
    #img[img<(30/255.0)] = 0;
    #img[img>(127/255.0)] = 1;
    
    # g(I)
    #gI = morphsnakes.gborders(img, alpha=1000, sigma=5.48)
    gI = morphsnakes.gborders(img, alpha=800, sigma=10)
    
    # Morphological GAC. Initialization of the level-set.
    #AM: mgac = morphsnakes.MorphGAC(gI, smoothing=1, threshold=0.31, balloon=1)
    mgac = morphsnakes.MorphGAC(gI, smoothing=1, threshold=0.50, balloon=1)
    #mgac.levelset = circle_levelset(img.shape, (100, 126), 20)
    
    #CasoAs.bmp
    #mgac.levelset = circle_levelset(img.shape, (230, 330), 20)
    #CasoAs.bmp
    iniPoint = getInitialPoint(img_source)
    #iniPoint = getInitialPoint(img)
    iniPointX = iniPoint[0]
    iniPointY = iniPoint[1]
    print('-> iniPoint[0]: '+ str(iniPointX))
    print('-> iniPoint[1]: '+ str(iniPointY))

    mgac.levelset = circle_levelset(img.shape, (iniPointY, iniPointX), 30)

    # Visual evolution.
    ppl.figure()
    #morphsnakes.evolve_visual(mgac, num_iters=45, background=img)
    morphsnakes.evolve_visual(mgac, num_iters=45, background=img)    

if __name__ == '__main__':
    print("""""")

    img_path = "../../base de imagens/Ecocardiograma/Casos/Todos os Casos/"
    #img_name = "CasoID"
    #img_name = "CasoDD"
    #img_name = "CasoQD"
    img_name = [
        "CasoAS.bmp", "CasoAD.bmp", "CasoBD.bmp", "CasoBS.bmp", "CasoCD.bmp", "CasoCS.bmp", "CasoDD.bmp",   #Ini. index: 0
        "CasoDS.bmp", "CasoED.bmp", "CasoES.bmp", "CasoFD.bmp", "CasoFS.bmp", "CasoGD.bmp", "CasoGS.bmp",   #Ini. index: 7
        "CasoHD.bmp", "CasoHS.bmp", "CasoID.bmp", "CasoIS.bmp", "CasoJD.bmp", "CasoJS.bmp", "CasoLD.bmp",   #Ini. index: 14
        "CasoLS.bmp", "CasoKD.bmp", "CasoKS.bmp", "CasoMD.bmp", "CasoMS.bmp", "CasoND.bmp", "CasoNS.bmp",   #Ini. index: 21
        "CasoPD.bmp", "CasoPS.bmp", "CasoQD.bmp", "CasoQS.bmp", "CasoOD.bmp", "CasoOS.bmp"]                 #Ini. index: 28

    img_source = img_path+'/'+img_name[1]
    img = imread(img_source)[...,0]/255.0

    test_nodule(img)
    #AM: test_starfish()
    #AM: test_lakes()
    #AM: test_confocal3d()
    ppl.show()