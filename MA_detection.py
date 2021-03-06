# -*- coding: utf-8 -*-
"""
Created on Wed Jan 03 00:39:11 2018
@author: David Marín Del Pino
@author: Jenifer Rodriguez Casas
Grado Ingeniería en Telemática
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
    return skimage.img_as_ubyte(result)
    
# Construcción del histograma
def makeHist(img, title):
    plt.hist(img.ravel(),bins = 50)
    plt.title(title)
    plt.xlabel("Intensidad de gris")
    plt.ylabel("pixels")
    
def getBvMask(img):
    # Extracción de componentes
    b,g,r = cv2.split(img) 
    cv2.imwrite("canal_verde.jpg", g)
    #showImage(g)
    
    # Ecualización de histograma mediante filtro adaptativo.
    # El objetivo es mejorar el contraste para extraer la máscara del árbol vascular.
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(6,6))
    enhanced_contrast_img = clahe.apply(g)
    cv2.imwrite("verde_contraste_mejorado.jpg", enhanced_contrast_img)
    #showImage(enhanced_contrast_img)
    
    
    # Tres pasadas de apertura y cierre para implementar un tophat dual. 
    # Aplicando la función correspondiente de opencv directamente el resultado no es el esperado
    r1 = cv2.morphologyEx(enhanced_contrast_img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
    R1 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
    r2 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
    R2 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
    r3 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)
    R3 = cv2.morphologyEx(r3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)	
    
    # tophat dual: diferencia entre el cierre y la imagen original.
    f4 = cv2.subtract(R3,enhanced_contrast_img)
    f5 = clahe.apply(f4)		
    cv2.imwrite("tophat.jpg", f5)
    
    # Eliminación del ruido perimetral
    ret,f6 = cv2.threshold(f5,15,255,cv2.THRESH_BINARY)	
    mask = np.ones(f5.shape[:2], dtype="uint8") * 255	
    im2, contours, hierarchy = cv2.findContours(f6.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
    	if cv2.contourArea(cnt) <= 200:
    		cv2.drawContours(mask, [cnt], -1, 0, -1)			
    im = cv2.bitwise_and(f5, f5, mask=mask)
    cv2.imwrite("tophat_sin_ruido.jpg", im)
    
    # Umbralización y erosión para recuperar los vasos sanguíneos
    ret,fin = cv2.threshold(im,15,255,cv2.THRESH_BINARY_INV)			
    newfin = cv2.erode(fin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)	
    cv2.imwrite("arbol_vascular.jpg", newfin)
    
    # Eliminación de elementos curvos
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
    cv2.imwrite("arbol_vascular_mejorado.jpg", finimage)	
    blood_vessels = finimage#cv2.bitwise_not(finimage)
    cv2.imwrite("mascara_arbol_vascular.jpg", blood_vessels)
    return blood_vessels

# Escribir el nombre de la carpeta y de la imagen a tratar
path = './ImagesTIM/'
img_folder = 'MAE0000043/'
img_name = 'DS000DGS.JPG'
img = io.imread(path + img_folder + img_name)
showImage(img)
cv2.imwrite("original.jpg", img)

# Extracción de máscara vascular
bv_mask = getBvMask(img)

# Inpainting sobre la imagen original para extraer los vasos:
inpaint=cv2.inpaint(img, bv_mask,8,cv2.INPAINT_TELEA)
io.imsave('sin_vasos.jpg', inpaint)

# Componente verde de la imagen sin vasos
ri,inpaint_g,bi = cv2.split(inpaint)

# Realce de contraste con ecualización CLAHE
clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(6,6))
enhanced_contrast_inpaint = clahe.apply(inpaint_g)
showImage(enhanced_contrast_inpaint)

# filtro de medianas para eliminar el ruido
from skimage.morphology import square
from skimage.filters import median
kernel = square(3)
inpaint_g_fm = median(enhanced_contrast_inpaint, kernel)
showImage(inpaint_g_fm)
io.imsave('base_para_deteccion_filtro_medianas.jpg', inpaint_g_fm)

# filtro gaussiano
from skimage.filters import gaussian
inpaint_g_fg = skimage.img_as_ubyte(gaussian(enhanced_contrast_inpaint, 2))
showImage(inpaint_g_fg)
io.imsave('base_para_deteccion_filtro_Gauss.jpg', inpaint_g_fg)

# El resultado del filtro de medianas está más contrastado. 
# Vamos a hacer el tophat dual con esa imagen

# Máscara externa
r,g,b = cv2.split(img)
em1 = umbralize(r, skimage.filters.threshold_otsu(r))
ext_mask = cv2.bitwise_not(em1)

# tophat dual
kernel_thd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
thd = cv2.morphologyEx(inpaint_g_fm,cv2.MORPH_BLACKHAT,kernel_thd)
thd_clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(5,5))
thd_eh = thd_clahe.apply(thd)
candidates = cv2.subtract(thd_eh, ext_mask)
#candidates = cv2.subtract(candidates, bv_mask)
showImage(candidates)

# Umbralización
# Es importante encontrar un umbral adecuado. Salen demasiados candidatos
th_cand = umbralize(candidates, 28)
showImage(th_cand)
io.imsave('mascara de cadidatos.jpg', th_cand)
im_labeled, n_labels = skimage.measure.label(th_cand, 8,0, True)
im_props = skimage.measure.regionprops(im_labeled)
#para remover objetos pequeños
im_small = skimage.morphology.remove_small_objects(im_labeled,3)
# Creacion del disco 
kernel = skimage.morphology.disk(1)
#Aplicacion del filtrado mediante máscara
#im_opened = skimage.morphology.binary_dilation(im_labeled, kernel)
#im_opened = skimage.img_as_ubyte(im_opened)
im_opened = skimage.morphology.binary_dilation(im_small, kernel)
im_opened = skimage.img_as_ubyte(im_opened)
#showImage(im_opened)
# Visualice la imagen filtrada y compare
#io.imshow(im_opened)
#http://scikit-image.org/docs/dev/api/skimage.morphology.html#skimage.morphology.remove_small_objects
io.imsave('candidatos_definitivos.jpg', im_opened)
