# -*- coding: utf-8 -*-
"""
Created on Wed Jan 03 00:39:11 2018

@author: David Marín Del Pino
"""

from skimage import io
from skimage import filter as skfilt
from skimage import img_as_float
from skimage import exposure
import matplotlib.pyplot as plt
import numpy as np

# Mostrar imagen
def showImage(img):
    io.imshow(img)
    plt.axis('off')
    
# Negativizar la imagen    
def negative(img):
    pixmat = list(img.shape)
    negative = np.zeros(pixmat)
    for i in range (0,pixmat[0]-1):
        for j in range (0, pixmat[1]-1):
            value = 255 - img[i,j]
            negative[i,j] = value
    return negative

# Obtener imagen umbralizada
def umbralize(img, threshold):
    pixmat = list(img.shape)
    result = np.zeros(pixmat)
    for i in range (0,pixmat[0]-1):
        for j in range (0, pixmat[1]-1):
            value = img[i,j]
            if value < threshold:
                result[i,j] = 0
            else:
                result[i,j] = 1   
    return result
    
# Construcción del histograma
def makeHist(img, title):
    plt.hist(img.ravel(),bins = 50)
    plt.title(title)
    plt.xlabel("Intensidad de gris")
    plt.ylabel("pixels")

# Escribir el nombre de la carpeta y de la imagen a tratar
path = './ImagesTIM/'
img_folder = 'MAE0000043/'
img_name = 'DS000DGS.JPG'
img = io.imread(path + img_folder + img_name)
showImage(img)
# Extracción de la componente verde
g_comp = img[:,:,1]
showImage(g_comp)
# Negativizado de la componente verde
g_comp_neg = negative(g_comp)
showImage(g_comp_neg)
#pendiente de implementar: mejora de contraste mediante filtro predictivo 2D
#Imprescindible para poder segmentar y eliminar el árbol vascular

# Umbralización para obtener el arbol vascular
th = skfilt.threshold_otsu(g_comp_neg)
bv_mask = umbralize(g_comp_neg, th)
showImage(bv_mask)