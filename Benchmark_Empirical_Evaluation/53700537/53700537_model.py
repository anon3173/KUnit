
import keras
from keras.layers import GlobalAveragePooling1D
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
        model.add(Conv1D(20, 20, activation='relu', input_shape=(1000, 1)))
        model.add(MaxPooling1D(3))
        model.add(Conv1D(20, 10, activation='relu'))
        model.add(MaxPooling1D(3))
        model.add(Conv1D(20, 10, activation='relu'))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(3, activation='relu', use_bias=False))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
       
