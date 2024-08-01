

from tensorflow.keras.optimizers import SGD, Adam, RMSprop
import tensorflow 
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Activation, Dropout,Conv1D, MaxPooling1D, Flatten, InputLayer, BatchNormalization
class MyModel(tensorflow.keras.layers.Layer):
     
    def __init__(self):
        super(MyModel, self).__init__()
        
    def call(self):
        n_feats = 2
        model = Sequential()
        model.add(Dense(units=2, activation='sigmoid', input_shape=(n_feats,)))
        model.add(Dense(units=n_feats, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
              metrics=['accuracy'])
        return model
       
    
