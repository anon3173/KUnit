from keras.models import Sequential
from keras.layers import Dense
import numpy,pandas
import sklearn.datasets
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
numpy.random.seed(7)

# -----------------------------------------------------------------------------
# generate sinus.
# -----------------------------------------------------------------------------

T = 1000
X = numpy.array(range(T))
Y = numpy.sin(3.5 * numpy.pi * X / T) 


# -----------------------------------------------------------------------------
# Draw training data
# -----------------------------------------------------------------------------

# plt.scatter(X, Y, s = 10)
# plt.show()

# -----------------------------------------------------------------------------
# Build Keras model (my keras uses TensorFlow backend)
# -----------------------------------------------------------------------------

input_dim = 1

model = Sequential()
model.add(Dense(10, input_dim = input_dim, activation='tanh'))
model.add(Dense(90, activation='tanh'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(1,activation='tanh'))


# -----------------------------------------------------------------------------
# Comile and fit
# -----------------------------------------------------------------------------

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=50, batch_size=10)