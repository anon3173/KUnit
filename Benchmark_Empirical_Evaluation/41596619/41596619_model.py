

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
        model.add(Dense(3, input_dim=3, kernel_initializer='uniform', activation='relu'))
        model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

        # Compile model
        sgd = SGD(lr=0.001)
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
        return model
    

    