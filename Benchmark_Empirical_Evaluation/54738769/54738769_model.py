
import keras
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
import tensorflow 
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Activation, Dropout,Conv1D, MaxPooling1D, Flatten, InputLayer, BatchNormalization, GlobalAveragePooling1D
class MyModel(tensorflow.keras.layers.Layer):
     
    def __init__(self):
        super(MyModel, self).__init__()
        
    def call(self):
        model = Sequential()
        model.add(Dense(18, input_dim=18, activation='tanh'))
        model.add(Dense(36, activation='relu'))
        model.add(Dense(72, activation='relu'))
        model.add(Dense(72, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='softmax'))

        # Compile model

        model.compile(loss='mean_squared_error', optimizer='adam')
        return model
       

    