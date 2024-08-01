
import numpy as np
from keras.layers import Dense, Activation
from keras.models import Sequential
import matplotlib.pyplot as plt
import math
import time
import sys
import keras
from sklearn.preprocessing import  StandardScaler


x = np.arange(-100, 100, 0.5)
y = x**4

x_train = x.reshape(400,1)

model = Sequential()
model.add(Dense(50, input_shape=(1,)))
model.add(Activation('sigmoid'))
model.add(Dense(50) )
model.add(Activation('elu'))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')


model.fit(x, y, epochs=1000, batch_size=len(x), verbose=1,)
 