

from tensorflow.keras.optimizers import SGD, Adam, RMSprop
import tensorflow 
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Activation, Dropout,Conv1D, MaxPooling1D, Flatten, InputLayer, BatchNormalization
class MyModel(tensorflow.keras.layers.Layer):
     
    def __init__(self):
        super(MyModel, self).__init__()
        
    def call(self):
        #Design your model here and return the model
        input_dim = 1

        model = Sequential()
        model.add(Dense(10, input_dim = input_dim, activation='tanh'))
        model.add(Dense(90, activation='tanh'))
        model.add(Dense(10, activation='tanh'))
        model.add(Dense(10, activation='tanh'))
        model.add(Dense(10, activation='tanh'))
        model.add(Dense(10, activation='tanh'))
        model.add(Dense(10, activation='tanh'))
        model.add(Dense(10, activation='tanh'))
        model.add(Dense(10, activation='tanh'))
        model.add(Dense(10, activation='tanh'))
        model.add(Dense(10, activation='tanh'))
        model.add(Dense(1,activation='tanh'))

        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        return model
       
 
    