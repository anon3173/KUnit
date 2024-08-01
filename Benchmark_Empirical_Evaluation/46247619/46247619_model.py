
import keras
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
        model.add(Conv1D(30, kernel_size=3, activation='relu', input_shape=(50, 1)))
        model.add(Conv1D(40, kernel_size=3, activation='relu'))
        model.add(Conv1D(120, kernel_size=3, activation='relu'))
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='relu'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam())
        return model
      
