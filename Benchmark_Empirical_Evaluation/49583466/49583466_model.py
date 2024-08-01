
import keras
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
import tensorflow 
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Activation, Dropout,Conv1D, MaxPooling1D, Flatten, InputLayer, BatchNormalization
class MyModel(tensorflow.keras.layers.Layer):
     
    def __init__(self):
        super(MyModel, self).__init__()
        
    def call(self):
        regressor = Sequential()
        regressor.add(Dense(units=20, activation='sigmoid', kernel_initializer='uniform', input_dim=1))
        regressor.add(Dense(units=20, activation='sigmoid', kernel_initializer='uniform'))
        regressor.add(Dense(units=20, activation='sigmoid', kernel_initializer='uniform'))
        regressor.add(Dense(units=1))
        regressor.compile(loss='mean_squared_error', optimizer='sgd')
        return regressor
      
