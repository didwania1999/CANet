# -*- coding: utf-8 -*-
"""CANetDME_trained.ipynb


"""

from google.colab import drive
drive.mount('/content/drive')

from matplotlib import pyplot as plt
import numpy as np
import os
import time
import keras
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array,load_img
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
from keras.layers import Dense, Activation, Flatten
from keras.layers import merge, Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import math

import pandas as pd  
train = pd.read_csv(r"/content/drive/MyDrive/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv")
test = pd.read_csv(r"/content/drive/MyDrive/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv")

train['Riskofmacularedema'] = train['Risk of macular edema '].astype(str)

def append_ext(fn):
    return fn+".jpg"
train["Imagename"]=train["Image name"].apply(append_ext)
test["Imagename"]=test["Image name"].apply(append_ext)

train

test

print(train.Riskofmacularedema.value_counts())
a=train.Riskofmacularedema.value_counts().plot(kind="bar")

train_folder = "/content/drive/MyDrive/B. Disease Grading/1. Original Images/a. Training Set"
test_folder = "/content/drive/MyDrive/B. Disease Grading/1. Original Images/b. Testing Set"

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Define your data generator
train_gen = ImageDataGenerator(
rotation_range=45,
rescale=1./285,
horizontal_flip=False
)
test_gen = ImageDataGenerator(rescale = 1.285)

train_data = train_gen.flow_from_dataframe(dataframe = train, 
directory = train_folder, x_col = "Image name", 
y_col = "Risk of macular edema ", seed = 42,
batch_size = 32, shuffle = True, validate_filenames=False,
class_mode="categorical",target_size = (224, 224))

test_data = test_gen.flow_from_dataframe(dataframe = test, 
directory = test_folder, x_col = "Image name",validate_filenames=False, 
y_col = "Risk of macular edema ",
batch_size = 32, shuffle = False, 
class_mode=None,target_size =  (224, 224))

class_names = train_data.classes
classes=list(set(class_names))
print(class_names)
NUM_CLASSES = len(class_names)
NUM_CLASSES

classes

plt.figure(figsize=(10, 10))
for i in range(0,9):
  plt.imshow(images[i])
  plt.axis("off")

from tensorflow.keras.models import Model
from keras import  models
# ResNet as Feature Extractor
inputTensor = Input(shape=(224,224,3,))
resnet=tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=inputTensor,pooling=None)
F=resnet.layers[-1].output
#Fcmax
X1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1,1), padding='valid')(F)
X1 = tf.keras.layers.Flatten()(X1)
X1 = tf.keras.layers.Reshape((1024,72))(X1)
X1=tf.keras.activations.sigmoid(
    X1
)
#Fcavg
X2 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(1,1), padding='valid')(resnet.layers[-1].output)
X2 = tf.keras.layers.Flatten()(X2)
X2 = tf.keras.layers.Reshape((1024,72))(X2)
X2=tf.keras.activations.sigmoid(
    X2
)
#Ac
X = tf.keras.layers.add([X1, X2])
X = tf.keras.layers.Reshape((1024,72))(X2)
#Ac*F

a=Dense(98)(X)
a=tf.keras.layers.Reshape((1024,98))(a)
b=Dense(98)(F)
b=tf.keras.layers.Reshape((1024,98))(F)
Fi=tf.keras.layers.multiply(
    [a,b]
)
Fi=Dense(64)(Fi)
Fi=tf.keras.layers.Reshape((8,8,1024))(Fi)
#AvgPool Fiavg
Fiavg = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(1,1), padding='valid')(Fi)
#Fiavg = tf.keras.layers.Reshape((1024,72))(X2)

#Maxpool Fimax
Fimax = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1,1), padding='valid')(Fi)
#Fimax = tf.keras.layers.Reshape((1024,72))(X1)

As = tf.keras.layers.concatenate([Fimax, Fiavg ])
As=tf.keras.layers.Reshape((1024,98))(a)
As = tf.keras.layers.Conv1DTranspose(filters=64, kernel_size=3,padding="same", activation='sigmoid')(As)

Fi=tf.keras.layers.Reshape((1024,64))(Fi)

#As*Fi
Fii=tf.keras.layers.multiply(
    [As,Fi]
)

#============Done till here!====================Disease Dependent Module=========

# Fii = tf.keras.layers.Reshape((8,8,1024))(Fii)

# X1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1,1), padding='valid')(Fii)
# X1 = tf.keras.layers.Reshape((1024,49))(X1)

# X2 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(1,1), padding='valid')(Fii)
# X2 = tf.keras.layers.Reshape((1024,49))(X2)
#======================================

Fii=tf.keras.layers.Flatten()(Fii)

X = Dense(3,activation="softmax")(Fii)
model = models.Model(inputs=inputTensor, outputs=X)
model.summary()

# model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
# hist = model.fit(train_data,epochs=20)

y_predicts = model.predict(test_data)
print(y_predicts)
y_predicts.shape

images = image.load_img("/content/drive/MyDrive/B. Disease Grading/1. Original Images/b. Testing Set/IDRiD_011.jpg", target_size=( 224, 224))    
x = image.img_to_array(images)
x = x.reshape(1,x.shape[0],x.shape[1],x.shape[2])
prediction=model.predict(x)
print(prediction)
print(np.argmax(prediction))
plt.figure(figsize=(5, 5))
plt.imshow(x[0].astype("uint8"))
plt.axis("off")

images = image.load_img("/content/drive/MyDrive/B. Disease Grading/1. Original Images/a. Training Set/IDRiD_001.jpg", target_size=( 224, 224))    
x = image.img_to_array(images)
x = x.reshape(1,x.shape[0],x.shape[1],x.shape[2])
prediction=model.predict(x)
print(prediction)
print(np.argmax(prediction))
plt.figure(figsize=(10, 10))
plt.imshow(x[0].astype("uint8"))
plt.axis("off")

images = image.load_img("/content/drive/MyDrive/B. Disease Grading/1. Original Images/b. Testing Set/IDRiD_102.jpg", target_size=( 224, 224))    
x = image.img_to_array(images)
x = x.reshape(1,x.shape[0],x.shape[1],x.shape[2])
prediction=model.predict(x)
print(prediction)

print(np.argmax(prediction))
plt.figure(figsize=(10, 10))
plt.imshow(x[0].astype("uint8"))
plt.axis("off")
#should have been classified as 2

#Yes

