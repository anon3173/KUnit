
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
        model.add(Dense(1, activation = 'linear', input_dim = 1))
   
        model.compile(optimizer = 'rmsprop', loss = 'mean_squared_error', metrics = ['accuracy'])
        return model
