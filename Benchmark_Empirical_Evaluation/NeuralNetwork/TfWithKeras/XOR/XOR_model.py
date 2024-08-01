
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
        model.add(Dense(units=8, input_dim=2))
        model.add(Activation('tanh'))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
    
        model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.1), metrics=['accuracy'])
        return model
