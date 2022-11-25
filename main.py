from configparser import Interpolation
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
from keras.models import load_model
import pickle


#Loading MNIST dataset
(xtrain, ytrain), (xtest, ytest) = tf.keras.datasets.mnist.load_data()

#scaling
xtrain = xtrain/255
xtest = xtest/255

# print(xtrain.shape)
# print(xtest.shape)

#flatten the data
x_train = xtrain.reshape(len(xtrain),28*28)
x_test = xtest.reshape(len(xtest),28*28)

#create neural network
model = keras.Sequential([
        keras.layers.Dense(100, input_shape=(28*28,) ,activation='sigmoid'), #to improve accuracy
        keras.layers.Dense(10,activation='sigmoid')

        ])


model.compile(optimizer= 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'] )

model.fit(x_train , ytrain , epochs = 10)
pickle.dump(model, open('model.pkl', 'wb'), protocol=2)
with open('model_pickle','wb') as file:
    pickle.dump(model,file)
print('saved')

#evaluate test set
acc  = model.evaluate(x_test , ytest)[1]
print('accuracy ' , acc , '%')

# predict an example 
print(xtest[0].shape)
plt.imshow(xtest[0],interpolation='nearest')
plt.show()

y_predict = model.predict(x_test)
print(np.argmax(y_predict[0]))


# #save the trained model
# model.save('model.h5')
# print('done')
