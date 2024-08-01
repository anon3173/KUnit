

from tensorflow.keras.optimizers import SGD, Adam, RMSprop
import tensorflow 
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Activation, Dropout,Conv1D, MaxPooling1D, Flatten, InputLayer, BatchNormalization
class MyModel(tensorflow.keras.layers.Layer):
     
    def __init__(self):
        super(MyModel, self).__init__()
        
    def call(self):
        model = Sequential()
        model.add(Dense(50, input_dim=1))
        model.add(Activation('relu'))
        model.add(Dense(30, kernel_initializer='uniform'))
        model.add(Activation('relu'))
        model.add(Dense(1))
        model.add(Activation('linear'))
        #training
        sgd = SGD(lr=0.1)
        model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])
        return model
       
 