
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
        model.add(Dense(5, activation='sigmoid',input_dim=5))
        model.add(Dense(4, activation='sigmoid'))
        model.add(Dense(1, activation='sigmoid'))

        sgd = SGD(lr=0.1)
        model.compile(optimizer= sgd, loss='mse')
        return model
