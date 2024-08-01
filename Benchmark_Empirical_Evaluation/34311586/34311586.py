from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
import numpy
import keras
import time

X = numpy.array([[1., 1.], [0., 0.], [1., 0.], [0., 1.], [1., 1.], [0., 0.]])
y = numpy.array([[0.], [0.], [1.], [1.], [0.], [0.]])
model = Sequential()
model.add(Dense(2, input_dim=2, kernel_initializer='uniform'))
model.add(Activation('sigmoid'))
model.add(Dense(3, kernel_initializer='uniform'))
model.add(Activation('sigmoid'))
model.add(Dense(1,kernel_initializer='uniform'))
model.add(Activation('softmax'))
start_time =time.clock()
sgd = SGD(lr=0.001, decay=1e-6)
model.compile(loss='mean_squared_error', optimizer=sgd)

model.fit(X, y, nb_epoch=20 ,)   
        