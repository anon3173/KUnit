
import keras
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
import tensorflow 

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
class MyModel(tensorflow.keras.layers.Layer):
     
    def __init__(self):
        super(MyModel, self).__init__()
        
    def call(self):
        model = Sequential()
        model.add(Dense(2, input_dim=2))
        model.add(Dense(1))
        model.compile(optimizer='rmsprop', loss='mse')
        return model
 