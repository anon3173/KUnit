
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
        
        model_05_01 = Sequential()
        model_05_01.add(Conv1D(filters=16, kernel_size=12, input_shape=(28*28, 1)))
        model_05_01.add(MaxPooling1D(pool_size=4))

        model_05_01.add(Conv1D(filters=32, kernel_size=12))
        model_05_01.add(MaxPooling1D(pool_size=4))

        model_05_01.add(Conv1D(filters=16, kernel_size=12))
        model_05_01.add(MaxPooling1D(pool_size=4))

        model_05_01.add(Flatten())

        model_05_01.add(Dense(16, activation='relu'))
        model_05_01.add(Dense(2, activation='sigmoid'))

        model_05_01.compile(loss='logcosh', optimizer='adam', metrics=['accuracy'])
        return model_05_01
 