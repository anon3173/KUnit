

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
        model = Sequential()
        model.add(Dense(12, input_dim=11, kernel_initializer='normal'))
        model.add(Activation('relu'))
        model.add(Dense(12, kernel_initializer='normal'))
        model.add(Activation('relu'))
        model.add(Dense(1, kernel_initializer='normal'))
        model.add(Activation('relu'))

        model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
        return model
       
    

    