
import keras
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
import tensorflow 

import numpy as np
from tensorflow.keras.models import Sequential
import tensorflow.keras.activations as tfnn
from tensorflow.keras.layers import Dense , Activation, Dropout,Conv1D, MaxPooling1D, Flatten, InputLayer, BatchNormalization, GlobalAveragePooling1D
class MyModel(tensorflow.keras.layers.Layer):
     
    def __init__(self):
        super(MyModel, self).__init__()
        
    def call(self):
        # create model
        model = Sequential()
        model.add(Dense(input_dim=3, units=1))
        model.add(Activation('linear'))
        rmsprop = RMSprop(lr=1e-10)
        model.compile(loss='mse', optimizer=rmsprop,  metrics=['accuracy'])
        return model
