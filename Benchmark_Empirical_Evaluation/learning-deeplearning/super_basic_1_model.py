
import keras
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
import tensorflow 

import numpy as np
from tensorflow.keras.models import Sequential
import tensorflow.keras.activations as tfnn
from tensorflow.keras.layers import Dense , Activation, Dropout,Conv1D, MaxPooling1D, Flatten, InputLayer, BatchNormalization, GlobalAveragePooling1D
class MyModel(tensorflow.keras.layers.Layer):
     
    def __init__(self):
        super(MyModel, self).__init__()
        
         
    def call(self):
        # create model
        model = Sequential([
        Dense(16, input_shape=(1,), activation='relu'),
        Dense(32, activation='relu'),
        Dense(2, activation='softmax')
    ])
        model.compile(Adam(lr=100), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

 
    