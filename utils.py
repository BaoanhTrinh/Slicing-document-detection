from scipy.fftpack import dct, idct
import cv2
from scipy import ndimage, misc
from skimage import feature, filters
import numpy as np
from math import log10, sqrt
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy.ma as ma
import glob
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten,MaxPooling1D,Dropout
import tensorflow as tf
from tensorflow.keras import optimizers

NUM_FEATURES = 75
EPOCHS = 100
BATCH_SIZE = 8
lr = 1e-4
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

def get_features(PATH):
    # read lena RGB image and convert to grayscale
    rgb_im = cv2.imread(PATH)
    gray_im = cv2.cvtColor(rgb_im, cv2.COLOR_BGR2GRAY)
    dct_img = dct2 (gray_im)

    #Get positive coef.
    dct_positive_coef, dct_negative_coef = np.array(dct_img), np.array(dct_img)

    np.putmask(dct_positive_coef, dct_positive_coef < 0, 0)
    np.putmask(dct_negative_coef, dct_negative_coef > 0, 0)


    idct_positive_img = idct2(dct_positive_coef)
    idct_negative_img = idct2(dct_negative_coef)

    #Apply laplacian filter
    positive_laplacian_img = laplace (idct_positive_img)
    negative_laplacian_img = laplace (idct_negative_img)

    #Fuse image
    fuse_img = (np.array (positive_laplacian_img) + np.array (negative_laplacian_img)) / 2.0

    fuse_img = np.uint8(fuse_img) 

    # calculate residual image
    residual_img = np.abs (fuse_img - np.array (gray_im))

    hist_residual = hist_features(residual_img)
    hist_fuse = hist_features(fuse_img)
    hist_original = hist_features(gray_im)

    cv2.normalize(hist_residual,hist_residual,0,1,cv2.NORM_MINMAX)
    cv2.normalize(hist_fuse,hist_fuse,0,1,cv2.NORM_MINMAX)
    cv2.normalize(hist_original,hist_original,0,1,cv2.NORM_MINMAX)
    diff_hist = np.abs (np.array (hist_original) - np.array (hist_fuse))

    img_features = np.concatenate ((hist_residual, hist_fuse, diff_hist), axis = 0)
    
    return img_features
def load_data(folder):
	image_list =[]
	label_list = []
	for label in ['normal','slicing']:
    		for PATH in glob.glob(folder + r'\{}\*.jpg'.format(label)): 
		        features = get_features(PATH)
		        image_list.append(features)
		        if label == 'normal':
		            label_list.append(1)
		        else:
		            label_list.append(0)
	return image_list,label_list
def convert_data_to_tensor(image_list,label_list):
	import random
	temp = list(zip(image_list, label_list))
	random.shuffle(temp)
	image_list, label_list = zip(*temp)
	N = len(label_list)
	Y = np.asarray(label_list)
	X = np.zeros((N,NUM_FEATURES))
	for i in range(0,N):
	    X[i] = image_list[i].reshape(NUM_FEATURES,)
	return X,Y
def _predict(folder): 
	from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
	from sklearn import metrics
	image_list,label_list = load_data(folder)
	X,Y = convert_data_to_tensor(image_list,label_list)
	model.load_weights("model")
	N = len(label_list)
	X = X.reshape((N,n_features,1))
	y_pred = model.predict(X)
	#fpr, tpr, thresholds = metrics.roc_curve(Y,y_pred)
	#print('AUC = ',metrics.auc(fpr, tpr))
	return label_list,round(y_pred)
def build_model():
   	model = Sequential()
   	model.add(Conv1D(filters=32, kernel_size=3,padding='same', activation='relu', input_shape=(NUM_FEATURES,1)))
   	model.add(Conv1D(filters=128, kernel_size=3,padding = 'same', activation='relu'))
   	model.add(MaxPooling1D(strides =2,padding ='same'))
   	model.add(Conv1D(filters=32, kernel_size=3, padding='same',activation='relu'))
   	model.add(MaxPooling1D(strides =2,padding ='same'))
   	model.add(Flatten())
   	model.add(Dense(608,activation="relu"))
   	model.add(Dense(128,activation="relu"))
   	model.add(Dense(16,activation="relu"))
   	model.add(Dense(1, activation='sigmoid'))
   	model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=[keras.metrics.BinaryAccuracy(),
                           keras.metrics.AUC()])
   	return model