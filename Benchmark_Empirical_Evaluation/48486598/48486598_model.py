

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
        model.add(Dense(1, input_shape=(1,)))
        model.add(Dense(5))
        model.add(Dense(1, activation='linear'))

        # compile model
        model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
        return model
       
    
