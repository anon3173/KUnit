
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

        model.add(Dense(30, activation=tfnn.relu, input_dim=30))
        model.add(BatchNormalization(axis=1))

        model.add(Dense(60, activation=tfnn.relu))
        model.add(BatchNormalization(axis=1))

        model.add(Dense(1, activation=tfnn.softmax))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
       
