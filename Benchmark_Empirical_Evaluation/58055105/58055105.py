import numpy
import pandas
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import time
import sys
import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
# load the dataset
data = pandas.read_csv('../fixed.csv')
X = data.drop(['status', 'name'], axis = 1)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
Y = data['status']
# # transform the dataset
oversample = SMOTE()
X, Y = oversample.fit_resample(X, Y)

xTrain, xTest, yTrain, yTest = train_test_split(X, Y, train_size = 0.8)
model = Sequential()
model.add(Dense(48, input_shape=(22,), activation = 'relu'))
model.add(Dropout(0.5))
# model.add(Dense(1, activation = 'softmax'))
model.add(Dense(1, activation = 'sigmoid'))
optim = keras.optimizers.adam(lr=0.0001)
model.compile(optimizer = optim, loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(xTrain, yTrain, epochs = 20, batch_size = 5, validation_data = (xTest, yTest),verbose=1)