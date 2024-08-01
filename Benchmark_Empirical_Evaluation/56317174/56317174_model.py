

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
        model.add(BatchNormalization(input_shape=(4,)))
        model.add(Dense(4, activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(BatchNormalization())
        model.add(Dense(4, activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(BatchNormalization())
        model.add(Dense(4, activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=["accuracy"])
        return model
    

    