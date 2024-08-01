

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
        model = Sequential()
        model.add(Dense(50, input_dim=1, kernel_initializer = 'uniform', activation='relu'))
        model.add(Dense(50, kernel_initializer = 'uniform', activation='relu'))
        model.add(Dense(1, kernel_initializer = 'uniform', activation='sigmoid'))


        # Compile model
        opt = SGD(lr=0.01)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model

    