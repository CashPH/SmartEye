import tensorflow as tf
from keras import layer,models
import os
import numpy as np
import cv2
import random

#medidas de imagenes
width = 300
height= 300

route_train = 'hakc/test/'
route_predict= 'hakc/predict/'

train_x = []
train_y = []

for i in os.listdir(route_train):
    for j in os.listdir(route_train+i):
        img= cv2.imread(route_train+i+'/'+j)
        resized_image = cv2.resize(img,width,height)

        train_x.append(resized_image.shape)

        if i == 'armasb':
            train_y.append([0,1])
        else:
            train_y.append([1,0])
x_data = np.array(train_x)
y_data = np.array(train_y)

model = tf.keras.Sequential([
    layer.Conv2D(32, 3,3, input_shape=(width, height, 3)),
    layer.Activation('relu'),
    layer.MaxPooling2D(pool_size=(2,2)),
    layer.Conv2D(32, 3,3),
    layer.Activation('relu'),
    layer.MaxPooling2D(pool_size=(2,2)),
    layer.Conv2D(64, 3,3),
    layer.Activation('relu'),
    layer.MaxPooling2D(pool_size=(2,2)),
    layer.Flatten(),
    layer.Dense(64),
    layer.Activation('relu'),
    layer.Dropout(0.5),
    layer.Dense(2),
    layer.Activation('sigmoid')

])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 300

model.fit(x_data,y_data,epochs = epochs)

models.save_model(model, 'mimodelo.keras')