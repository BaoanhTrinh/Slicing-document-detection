from scipy.fftpack import dct, idct
import cv2
from scipy import ndimage, misc
from skimage import feature, filters
import numpy as np
from math import log10, sqrt
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr

# implement 2D DCT
def dct2(a):
    return dct(dct(a.T, norm=None).T, norm=None) #ortho

# implement 2D IDCT
def idct2(a):
    return idct(idct(a.T, norm=None).T, norm=None)   

def laplace (img):
	# # Apply Gaussian Blur
	blur = cv2.GaussianBlur(img,(5,5),0)
	# # Apply Laplacian operator in some higher datatype
	laplacian = cv2.Laplacian(blur,cv2.CV_64F, ksize=5)
	# dst = cv2.Laplacian(src, bit, ksize)
	# laplacian = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
	
	return laplacian

def candy(image):
	# Compute the Canny filter for two values of sigma
	# edges1 = feature.canny(image)
	edges1 = cv2.Canny(image,100,200)
	edges1 = np.uint8(edges1) 
	# edges2 = feature.canny(image, sigma=3)
	return edges1

def sobel (image):
	img_blur = cv2.GaussianBlur(image, (3,3), 0)
	sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
	sobelxy = np.uint8(sobelxy) 
	return sobelxy

def hist_features(image):
	# hist,bins = np.histogram(image.ravel(),25,[0,256])
	
	hist = cv2.calcHist([image],[0],None,[25],[0,256]) 
	return hist

def quality_features(image1, image2):
    mse1 = mse(image1, image2)
    
    psnr1 = psnr (image1, image2, data_range=None)
    (ssim1, diff) = compare_ssim(image1, image2, full=True)

    return psnr1, mse1, ssim1


