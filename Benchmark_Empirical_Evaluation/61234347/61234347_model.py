
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
        model.add(Dense(1000, activation='tanh', input_shape=(4,)))
        model.add(Dense(500, activation='tanh'))
        model.add(Dense(250, activation='tanh'))
        model.add(Dense(125, activation='tanh'))
        model.add(Dense(64, activation='tanh'))
        model.add(Dense(32, activation='tanh'))
        model.add(Dense(9, activation='tanh'))
        model.add(Dense(3, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    