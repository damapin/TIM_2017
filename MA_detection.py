# -*- coding: utf-8 -*-
"""
Created on Wed Jan 03 00:39:11 2018

@author: David Marín Del Pino
"""
import skimage
from skimage import io
from skimage import filters as skfilt
from skimage import img_as_float
from skimage import exposure
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Mostrar imagen
def showImage(img):
    io.imshow(img)
    plt.axis('off')
    
# Negativizar la imagen    
def negative(img):
    pixmat = list(img.shape)
    for i in range (0,pixmat[0]-1):
        for j in range (0, pixmat[1]-1):
            value = 255 - img[i,j]
            img[i,j] = value

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
    
def getBvMask(img):
    # Extracción de componentes
    b,g,r = cv2.split(img)
    #showImage(g)
    
    # Ecualización de histograma mediante filtro adaptativo.
    # El objetivo es mejorar el contraste para extraer la máscara del árbol vascular.
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(4,4))
    enhanced_contrast_img = clahe.apply(g)
    #showImage(enhanced_contrast_img)
    
    
    # Tres pasadas de apertura y cierre para implementar un tophat. Directamente el resultado no es el esperado
    r1 = cv2.morphologyEx(enhanced_contrast_img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
    R1 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
    r2 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
    R2 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
    r3 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)
    R3 = cv2.morphologyEx(r3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)	
    f4 = cv2.subtract(R3,enhanced_contrast_img)
    f5 = clahe.apply(f4)		
    
    # Eliminación del ruido perimetral
    ret,f6 = cv2.threshold(f5,15,255,cv2.THRESH_BINARY)	
    mask = np.ones(f5.shape[:2], dtype="uint8") * 255	
    im2, contours, hierarchy = cv2.findContours(f6.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
    	if cv2.contourArea(cnt) <= 200:
    		cv2.drawContours(mask, [cnt], -1, 0, -1)			
    im = cv2.bitwise_and(f5, f5, mask=mask)
    # Umbralización y erosión para recuperar los vasos sanguíneos
    ret,fin = cv2.threshold(im,15,255,cv2.THRESH_BINARY_INV)			
    newfin = cv2.erode(fin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)	
    
    # Eliminación de pequeñas ramas
    fundus_eroded = cv2.bitwise_not(newfin)	
    xmask = np.ones(img.shape[:2], dtype="uint8") * 255
    x1, xcontours, xhierarchy = cv2.findContours(fundus_eroded.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)	
    for cnt in xcontours:
    	shape = "unidentified"
    	peri = cv2.arcLength(cnt, True)
    	approx = cv2.approxPolyDP(cnt, 0.04 * peri, False)   				
    	if len(approx) > 4 and cv2.contourArea(cnt) <= 3000 and cv2.contourArea(cnt) >= 100:
    		shape = "circle"	
    	else:
    		shape = "veins"
    	if(shape=="circle"):
    		cv2.drawContours(xmask, [cnt], -1, 0, -1)	
    	
    finimage = cv2.bitwise_and(fundus_eroded,fundus_eroded,mask=xmask)	
    blood_vessels = cv2.bitwise_not(finimage)
    return blood_vessels

# Escribir el nombre de la carpeta y de la imagen a tratar
path = './ImagesTIM/'
img_folder = 'MAE0000043/'
img_name = 'DS000DGS.JPG'
img = io.imread(path + img_folder + img_name)
showImage(img)

bv_mask = getBvMask(img)





