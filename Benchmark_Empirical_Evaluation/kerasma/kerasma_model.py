
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
        # create model
        model = Sequential()
        model.add(Dense(12, input_dim=8, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
 