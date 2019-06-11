from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2
from keras.backend import tensorflow_backend as K
import os
import glob
import time
import keras
from matplotlib import pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K2


#IOU calc
iou_smooth=1.

#Unet ile plaka bulmak icin gerekli input size'ları
img_width, img_height = 256, 256

char_list = ["0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","R","S","T","U","V","Y","Z","X","W"]

#Unet icin gereken loss fonksyonu, kesisen alana gore loss hesaplar
def IOU_calc(y_true, y_pred):
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	
	return 2*(intersection + iou_smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + iou_smooth)

def IOU_calc_loss(y_true, y_pred):
	return 1-IOU_calc(y_true, y_pred)

#Plakadaki karakterlerin sıralanması icin, karakterleri en'lerine bakarak sıralar
def compareRectWidth(a,b):
	return a < b

# Unet modeli yukluyor, 
model_unet = load_model('../src/gumruk_unetGU002.h5',custom_objects={'IOU_calc_loss': IOU_calc_loss, 'IOU_calc': IOU_calc})

#CNN modelini yukluyor, karakter tanıma icin

# CNN modelinin input sizeları
img_rows, img_cols = 28, 28

batch_size = 128
num_classes = 35
epochs = 12

if K2.image_data_format() == 'channels_first':
    input_shape = (1, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 1)

model_cnn = Sequential()
model_cnn.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model_cnn.add(Conv2D(64, (3, 3), activation='relu'))
model_cnn.add(MaxPooling2D(pool_size=(2, 2)))
model_cnn.add(Dropout(0.25))
model_cnn.add(Flatten())
model_cnn.add(Dense(128, activation='relu'))
model_cnn.add(Dropout(0.5))
model_cnn.add(Dense(num_classes, activation='softmax'))

model_cnn.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model_cnn.load_weights('../src/mert_cnn.h5')


#unet ile plakayi bulup, return ediyor. input olarak image path alıyor
def getPlateImage(filepath):
	image = cv2.imread(filepath)
	plate = image
	originalImage = image

	##model icin gerekli input boyutunu hazirliyor.
	image = cv2.resize(image, (256, 256)).astype("float32")	
	image = np.expand_dims(image, axis=0)
	
	#prediction binary image dönüyor
	pred = model_unet.predict(image)
	pred = pred.reshape((256,256,1))
	pred = pred.astype(np.float32)

	pred = pred*255

	pred = cv2.resize(pred, (originalImage.shape[1], originalImage.shape[0]))
	pred=np.uint8(pred)

	#resimdeki en buyuk beyaz alanı alıp(plaka lokasyonu) kesiyor
	contours, hierarchy = cv2.findContours(pred,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	largestArea = 0
	for contour in contours:
		tArea = cv2.contourArea(contour)
		if tArea > largestArea:
			largestArea = tArea
			x,y,w,h = cv2.boundingRect(contour)

	if largestArea > 0:
		plate = originalImage[y:y+h,x:x+w]

	else:
		print("PLATE COULD NOT FOUND")

	return plate

#plaka resmini alıp 
def getPlateString(plate):
	grayPlate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
	roiList = []
	wList = []
	charList = []
	retval, binary = cv2.threshold(grayPlate, 30.0, 255.0, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	contours,hierarchy = cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	idx =0 

	plateStr = []
	for cnt in contours:
		idx += 1
		x,y,w,h = cv2.boundingRect(cnt)
		roi=plate[y:y+h,x:x+w]
		if w > 15 and h > 30 and w <100 and h< 100:
			roiList.append(roi)
			wList.append(x)

			#cv2.imwrite("/home/utku/Desktop/rois/" + str(idx) +".jpg", roi)
			#cv2.waitKey(100)
			#predict roi, resize may needed
			#roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

			roi = np.asarray(roi)
			roi = np.resize(roi, (28,28))

			if K2.image_data_format() == 'channels_first':
				roi = roi.reshape(roi.shape[0], 1, img_rows, img_cols)
				input_shape = (1, img_rows, img_cols)
			else:
				roi = roi.reshape(1, img_rows, img_cols, 1)


			#roi = np.resize(roi, (28,28,1))
			#roi = np.expand_dims(roi, axis=0)


			roi = roi/255
			pred = model_cnn.predict(roi)
			
			#get index
			print("pred: ", pred)
			predd = pred[0]
			char_idx = np.argmax(predd)
			#char_idx = np.where(predd == 1) ##1 olanın indexi
			plate_char = char_list[char_idx];
			#append result to plateStr, may map the predict to a char(BUT HOW)
			plateStr.append(plate_char)

			print("plate_char is: ", plate_char)
			#break

	#sorting from left to right
	charList = [x for _,x in sorted(zip(wList,plateStr))]

	return charList


#plate = getPlateImage("sampleplate.jpg")
#plateString = getPlateString(plate)

#if 'X' in plateString: plateString.remove('X')

#print("plateString: ", plateString)

