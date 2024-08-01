
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
        model.add(Dense(128, activation='relu', input_shape=(3, )))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(1, activation='relu'))
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        return model
 