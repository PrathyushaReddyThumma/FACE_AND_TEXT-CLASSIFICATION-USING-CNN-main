import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.datasets import mnist # We are using Keras's built-in mnist dataset
#from tensorflow.keras.models import Sequential # A linear stack of model layers
#from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from sklearn.model_selection import train_test_split
#import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
"""


#READ THE DATA
# DATA IS TAKEN FROM mnist DATA SET:
(trainX, trainY), (testX, testY)=mnist.load_data()
print(f"train_x column names: {list(range(trainX.shape[1]))}")
print(f"train_y column names: ['label']")
#test data
print(f"test_x column names: {list(range(testX.shape[1]))}")
print(f"test_y column names: ['label']")

#trainX,testX, trainY,testY = train_test_split(mnist.data, mnist.target, test_size=0.2, random_state=42)


#preprocessing of the data:

#1. Checking missing values:
print('Number of missing values in training set:', np.isnan(trainX).sum())
print('Number of missing values in test set:', np.isnan(testY).sum())

#2. Removing Duplicates:
trainX = trainX.reshape(trainX.shape[0], -1)
testX = testX.reshape(testX.shape[0], -1)
train = np.hstack((trainX, trainY.reshape(-1, 1)))
test = np.hstack((testX, testY.reshape(-1, 1)))
train = np.unique(train, axis=0)
test = np.unique(test, axis=0)
trainX, trainY = train[:, :-1], train[:, -1].astype(int)
testX, testY = test[:, :-1], test[:, -1].astype(int)
print("Test and Train values:", train, test)

#Check the Data Structure:
(trainX, trainY), (testX, testY) = mnist.load_data()

print("Training data shape:", trainX.shape)    # (60000, 28, 28)
print("Training label shape:", trainY.shape)  # (60000,)
print("Test data shape:", testX.shape)        # (10000, 28, 28)
print("Test label shape:", testY.shape)

# check for  some instances
print(f'\nA sample with associated label: {trainY[0]}')
plt.imshow(trainX[0])

#Creating a model:

shape_input=(28,28,1)
no_filters=8
size_filter=3
size_pool=2

model=Sequential([
    Conv2D(no_filters, size_filter, shape_input=shape_input),
    MaxPooling2D(size_pool=size_pool),
    Flatten(),
    Dense(10,activation='softmax'),


])
model.summary()

#Compiling the model:
optimizer= SGD
loss= 'binary_crossentropy'
metrics=['accuracy']

model.compile(
    optimizer,
    loss,
    metrics,
)
print("Model compilation Successful")


# Taining the model:
print('Reshaping the images to add channels ...')
trainX = np.expand_dims(trainX, axis=3)
testX = np.expand_dims(testX, axis=3)
print(f'Training data shape: {trainX.shape}')
print(f'Test data shape: {testX.shape}')

model.fit(
  trainX,
  to_categorical(trainY),
  epochs=numberOfEpochs,
  validation_data=(testX, to_categorical(testY))
)

# Predicting the model:

no_samples = 20

truthval = testY[:no_samples]
pred = model.predict(testX[:no_samples])

array_pred = np.argmax(pred, axis=1)

print('Expected: ', truthval)
print('Predicted: ', array_pred)
