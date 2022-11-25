from configparser import Interpolation
from pyexpat import model
from tensorflow import keras
import keras.api._v2.keras as keras 
from matplotlib import pyplot as plt
from matplotlib.cbook import flatten
import tensorflow as tf
from keras.models import Sequential
import numpy as np
import cv2 as cv
import os
from keras.datasets import mnist
from PIL import Image
import logging
logging.getLogger('tensorflow').disabled = True #to hide gpu error
import tensorflow as tf
from keras.callbacks import Callback
import matplotlib.image as image
from sklearn import model_selection, datasets
from sklearn.tree import DecisionTreeClassifier
import joblib
import pickle

#load the model

with open('model_pickle','rb') as file:
    model = pickle.load(file)
(xtrain, ytrain), (xtest, ytest) = tf.keras.datasets.mnist.load_data()

#scaling
xtrain = xtrain/255
xtest = xtest/255

# print(xtrain.shape)
# print(xtest.shape)

#flatten the data
x_test = xtest.reshape(len(xtest),28*28)

#check the accuracy
acc  = model.evaluate(x_test , ytest)[1]
print('accuracy ' , acc*100 , '%')

