
import keras
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
import tensorflow 
from keras.models import Model
import numpy as np
from tensorflow.keras.models import Sequential
import tensorflow.keras.activations as tfnn
from tensorflow.keras.layers import Dense , Activation, Dropout,Conv1D, MaxPooling1D, Flatten, InputLayer, BatchNormalization, GlobalAveragePooling1D
from keras.layers import Input, Subtract
class MyModel(tensorflow.keras.layers.Layer):
     
    def __init__(self):
        super(MyModel, self).__init__()
        
         
    def call(self):
        
        model = Sequential()
        model.add(Dense(19, activation='relu'))
        model.add(Dense(19, activation='relu'))
        model.add(Dense(19, activation='relu'))
        model.add(Dense(19, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model

    