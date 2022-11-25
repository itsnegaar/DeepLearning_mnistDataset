from tensorflow import keras
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
from keras.models import load_model
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten 
from keras import backend as k
from keras.layers import MaxPooling2D
from keras import models , datasets , layers
from keras import utils as np_utils
from keras.utils import to_categorical
from keras.utils.np_utils import to_categorical
import keras.utils
from keras import utils as np_utils


#Loading MNIST dataset
(xtrain, ytrain), (xtest, ytest) = tf.keras.datasets.mnist.load_data()

#scaling
xtrain = xtrain/255
xtest = xtest/255

xtrain = np.expand_dims(xtrain , axis = -1)
xtest = np.expand_dims(xtest , axis = -1)
# ytrain = to_categorical(ytrain)
# ytest = to_categorical(ytrain)
num_classes=10
y_train = keras.utils.to_categorical(ytrain, num_classes)
y_test = keras.utils.to_categorical(ytest, num_classes)
print(' x shape',xtrain.shape)
print(' y shape',ytrain.shape)



#create neural network
model = models.Sequential([
        layers.Conv2D(32 , (3,3) , activation = 'relu'),
        layers.MaxPooling2D((3,3)),
        layers.Conv2D(32 , (3,3) , activation = 'relu'),
        layers.MaxPooling2D(pool_size=(3, 3)),
        layers.Dense(100, input_shape=(28*28,) ,activation='relu'), #to improve accuracy
        layers.Dense(10,activation='softmax')

        ])

print('model--------------------')

model.compile(optimizer= 'adam',
              loss = 'categorical_crossentropy',
              metrics=['accuracy'] )

print('compile--------------------')

model.fit(xtrain, ytrain, epochs=12)
print('fit--------------------')


#evaluate test set
acc  = model.evaluate(xtest , ytest)[1]
print('accuracy ' , acc , '%')

# predict an example 
print(xtest[0].shape)
plt.imshow(xtest[0],interpolation='nearest')
plt.show()

y_predict = model.predict(xtest)
print(np.argmax(y_predict[0]))


# #save the trained model
# model.save('model.h5')
# print('done')
