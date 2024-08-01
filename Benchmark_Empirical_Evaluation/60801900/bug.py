from keras import Sequential
from keras.layers import Dense
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import MinMaxScaler

number_of_samples = 100
MAX_CONTEXTS = 430
X, y = make_regression(number_of_samples, 3)
scaler = MinMaxScaler(feature_range=(-1.9236537371711413, 1.9242677998256246))
X = scaler.fit_transform(X, y)

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(3, )))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='relu'))
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

history = model.fit(X, y, epochs=15, batch_size=2, verbose=1, shuffle=True, validation_split=0.2)
