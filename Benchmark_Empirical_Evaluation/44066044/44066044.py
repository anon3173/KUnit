
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn import preprocessing
import numpy
import os
import time 
import sys
import keras
from sklearn.preprocessing import StandardScaler
import pandas

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
seed = 4
numpy.random.seed(seed)

dataset = pandas.read_csv("../44066044/NetworkPackets.csv")
dataset = dataset.values
X = dataset[:, 0:11].astype(float)
Y = dataset[:, 11]

# sc= StandardScaler()
# X = sc.fit_transform(X)
batch_size = 5
print(X.shape)
print(Y.shape)

model = Sequential()
model.add(Dense(12, input_dim=11, kernel_initializer='normal'))
model.add(Activation('relu'))
model.add(Dense(12, kernel_initializer='normal'))
model.add(Activation('relu'))
model.add(Dense(1, kernel_initializer='normal'))
model.add(Activation('relu'))
# model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.fit(X, Y, nb_epoch=100, batch_size=5)
