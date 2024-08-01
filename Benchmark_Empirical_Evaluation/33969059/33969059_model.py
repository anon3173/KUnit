

from tensorflow.keras.optimizers import SGD, Adam, RMSprop
import tensorflow 
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Activation, Dropout,Conv1D, MaxPooling1D, Flatten, InputLayer, BatchNormalization
class MyModel(tensorflow.keras.layers.Layer):
     
    def __init__(self):
        super(MyModel, self).__init__()
        
    def call(self):
        #Design your model here and return the model
        INDIM = 3
        model = Sequential()
        model.add(Dense(INDIM, input_dim=INDIM, kernel_initializer='uniform'))
        model.add(Activation('tanh'))
        model.add(Dropout(0.5))
        model.add(Dense(2, kernel_initializer='uniform'))
        model.add(Activation('tanh'))
        model.add(Dropout(0.5))
        model.add(Dense(1, kernel_initializer='uniform'))
        model.add(Activation('softmax'))

        sgd = SGD(learning_rate=0.1)
    
        model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
        return model
    

    