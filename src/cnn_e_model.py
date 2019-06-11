from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import os
import glob
import cv2
import numpy as np

batch_size = 400
num_classes = 35
epochs = 2000

# input image dimensions
img_rows, img_cols = 28, 28


#ALL DATA
images = []
fileList = []
for filename in glob.glob('/home/issd/AI/sources/user-space/Utku/PLATE_CHARS/*.jpg'):
    fileList.append(filename)
    
fileList.sort()
fileList.sort(key=len)
    
for i in fileList:
    #print(i)
    img = cv2.imread(i)
    
    img = np.resize(img, (28,28))
    images.append(img)
    
#print(images)


#print(images[0].shape)

images = np.asarray(images)
#print(images.shape)
    
labels = np.empty((0,35))
word = ""
with open("text.txt", "r") as file:
    file.read(1)
    for line in file:
        for char in line:
            if char != " ":
                word += char
            if char == " ":
                word = ""   
        
        word = word.strip()
        if word != "": 
            labels = np.append(labels, [word])
        

x_train = images[:10000]
x_test = images[10000:10500]

y_train = labels[:10000]
y_test = labels[10000:10500]


print("ytrain: ", y_train)
print("ytest: ", y_test)

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print("x_train1 shape", x_train[0].shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

print("x_train is: ", x_train)
print("len of x_train is: ", len(x_train))
print("x_train[0] is: ", x_train[0])


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=None)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save_weights('mert_cnn.h5')

with open('mert_cnn_architecture.json', 'w') as f:
    f.write(model.to_json())

