# -*- coding: utf-8 -*-
"""
Created on Wed Jan 03 00:39:11 2018

@author: David Marín Del Pino
"""

from skimage import io
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
            if value > threshold:
                result[i,j] = 0
            else:
                result[i,j] = 1   
    return result

# Escribir el nombre de la imagen a tratar
img_name = 'MAE0000043'
path = './ImagesTIM/'
img = io.imread(path + img_name)
showImage(img)