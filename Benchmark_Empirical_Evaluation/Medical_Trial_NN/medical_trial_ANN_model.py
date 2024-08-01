
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
        
        model = Sequential([
            Dense(units=16, input_shape=(1,), activation='relu'),
            Dense(units=32, activation='relu'),
            Dense(units=2, activation='softmax')
        ])
        # call summary() to get a quick visualization
        model.summary()
        model.compile(optimizer=Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
 
    