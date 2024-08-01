from keras.models import Sequential
from keras.layers import Dense
import random
import numpy
from sklearn.preprocessing import MinMaxScaler

regressor = Sequential()
regressor.add(Dense(units=20, activation='sigmoid', kernel_initializer='uniform', input_dim=1))
regressor.add(Dense(units=20, activation='sigmoid', kernel_initializer='uniform'))
regressor.add(Dense(units=20, activation='sigmoid', kernel_initializer='uniform'))
regressor.add(Dense(units=1))
regressor.compile(loss='mean_squared_error', optimizer='sgd')

N = 5000
X = numpy.empty((N,))
Y = numpy.empty((N,))

for i in range(N):
    X[i] = random.uniform(-10, 10)
X = numpy.sort(X).reshape(-1, 1)

for i in range(N):
    Y[i] = numpy.sin(X[i])
Y = Y.reshape(-1, 1)

X_scaler = MinMaxScaler()
Y_scaler = MinMaxScaler()
X = X_scaler.fit_transform(X)
Y = Y_scaler.fit_transform(Y)

regressor.fit(X, Y, epochs=50, verbose=1, batch_size=32)


