
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
        model1 = keras.Sequential()
        model1.add(keras.layers.Dense(64, input_dim=14, activation='relu'))
        model1.add(keras.layers.Dense(128, activation='relu'))
        model1.add(keras.layers.Dense(64, activation='relu'))  
        model1.add(keras.layers.Dense(1, activation='softmax'))

        # compile the keras model
        model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model1
      
 