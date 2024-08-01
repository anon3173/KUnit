import keras.optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import numpy
from keras.optimizers import SGD, Adam
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# split into input (X) and output (Y) variables
X = numpy.array([[0.5, 1, 1], [0.9, 1, 2], [0.8, 0, 1], [0.3, 1, 1], [0.6, 1, 2], [0.4, 0, 1], [0.9, 1, 7], [0.5, 1, 4],
                 [0.1, 0, 1], [0.6, 1, 0], [1, 0, 0]])
y = numpy.array([[1], [1], [1], [2], [2], [2], [3], [3], [3], [4], [4]])

# create model
model = Sequential()
model.add(Dense(3, input_dim=3, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
# model.add(Dense(4, kernel_initializer='uniform', activation='softmax'))
# Compile model
sgd = SGD(lr=0.001)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
# model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
unit = int(model.layers[len(model.layers)-1].__getattribute__('units'))
# Fit the model
model.fit(X, y, epochs=150)
# calculate predictions
predictions = model.predict(X)
# round predictions
# rounded = [numpy.round(x) for x in predictions]
rounded = numpy.round(numpy.array(predictions))
print(rounded)
print('this is unit',type(y))