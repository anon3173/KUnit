
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
        model = Sequential()
        model.add(Dense(48, input_shape=(22,), activation = 'relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation = 'softmax'))

        optim = keras.optimizers.adam(lr=0.0001)
        model.compile(optimizer = optim, loss = 'binary_crossentropy', metrics = ['accuracy'])
        return model
