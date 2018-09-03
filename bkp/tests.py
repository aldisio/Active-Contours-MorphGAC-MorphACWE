
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

def test_nodule():
    # Load the image.
    #img = imread("testimages/mama07ORI.bmp")[...,0]/255.0
    #img_path = "testimages/CasoDD.bmp"
    img_path = "../../base de imagens/Ecocardiograma/Casos/Todos os Casos/"
    #img_name = "CasoID.bmp"
    img_name = "CasoQD.bmp"
    img_source = img_path+'/'+img_name
    
    img1 = imread(img_source)[...,0]/255.0
    
    #img[img <= (30/255)] = 0;
    #img[img >= (120/255)] = 1;
    img2 = cv2.imread(img_source,0)[...,0]/255.0
    img=img2


    print(type(img1))
    print(type(img2))
    cv2.waitKey(0)
    #img = imread("testimages/CasoDD.bmp")[...,0]/255.0
    #img = imread("testimages/CasoDS.bmp")[...,0]/255.0
    #img = imread("testimages/CasoBD.bmp")[...,0]/255.0  
    
    
    # g(I)
    #gI = morphsnakes.gborders(img, alpha=1000, sigma=5.48)
    gI = morphsnakes.gborders(img, alpha=1000, sigma=8)
    
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
    morphsnakes.evolve_visual(mgac, num_iters=50, background=img)    

def test_starfish():
    # Load the image.
    imgcolor = imread("testimages/seastar2.png")/255.0
    img = rgb2gray(imgcolor)
    
    # g(I)
    gI = morphsnakes.gborders(img, alpha=1000, sigma=2)
    
    # Morphological GAC. Initialization of the level-set.
    mgac = morphsnakes.MorphGAC(gI, smoothing=2, threshold=0.3, balloon=-1)
    mgac.levelset = circle_levelset(img.shape, (163, 137), 135, scalerow=0.75)
    
    # Visual evolution.
    ppl.figure()
    morphsnakes.evolve_visual(mgac, num_iters=110, background=imgcolor)

def test_lakes():
    # Load the image.
    imgcolor = imread("testimages/lakes3.jpg")/255.0
    img = rgb2gray(imgcolor)
    
    # MorphACWE does not need g(I)
    
    # Morphological ACWE. Initialization of the level-set.
    macwe = morphsnakes.MorphACWE(img, smoothing=3, lambda1=1, lambda2=1)
    macwe.levelset = circle_levelset(img.shape, (80, 170), 25)
    
    # Visual evolution.
    ppl.figure()
    morphsnakes.evolve_visual(macwe, num_iters=190, background=imgcolor)

def test_confocal3d():
    
    # Load the image.
    img = np.load("testimages/confocal.npy")
    
    # Morphological ACWE. Initialization of the level-set.
    macwe = morphsnakes.MorphACWE(img, smoothing=1, lambda1=1, lambda2=2)
    macwe.levelset = circle_levelset(img.shape, (30, 50, 80), 25)
    
    # Visual evolution.
    morphsnakes.evolve_visual3d(macwe, num_iters=200)

if __name__ == '__main__':
    print("""""")
    test_nodule()
    #AM: test_starfish()
    #AM: test_lakes()
    #AM: test_confocal3d()
    ppl.show()
