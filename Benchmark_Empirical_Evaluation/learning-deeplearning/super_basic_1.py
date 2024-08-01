import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy

import numpy as np
from random import randint
from sklearn.preprocessing import MinMaxScaler
import time 
import keras
# -> Start creating sample data

train_labels = []
train_samples = []

for i in range(50):
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    train_labels.append(1)

    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(0)

for i in range(1000):
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    train_labels.append(0)

    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(1)

# -> Preprocessing

train_labels = np.array(train_labels)
train_samples = np.array(train_samples)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_samples = scaler.fit_transform((train_samples).reshape(-1, 1))

# -> End creating sample data

model = Sequential([
    Dense(16, input_shape=(1,), activation='relu'),
    Dense(32, activation='relu'),
    # Dense(2, activation='softmax')
    Dense(1, activation='sigmoid')
])

# -> Alternate Syntax
# model = Sequential()
# model.add(Dense(5, input_shape=(3,)))
# model.add(Activation('relu'))

model.compile(Adam(lr=100), loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(Adam(lr=100), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.loss='sparse_categorical_crossentropy'
# model.loss

model.fit(scaled_train_samples, train_labels, validation_split=0.1, batch_size=10, epochs=20, shuffle=True, verbose=2)
print('Shape of train',scaled_train_samples.shape)
print('train labels',train_labels)

# # -> Create test data
# test_samples = []
# test_lables = []

# for i in range(10):
#     random_younger = randint(13, 64)
#     test_samples.append(random_younger)
#     test_lables.append(1)

#     random_older = randint(65, 100)
#     test_samples.append(random_older)
#     test_lables.append(0)

# for i in range(200):
#     random_younger = randint(13, 64)
#     test_samples.append(random_younger)
#     test_lables.append(0)

#     random_older = randint(65, 100)
#     test_samples.append(random_older)
#     test_lables.append(1)

# test_lables = np.array(train_labels)
# test_samples = np.array(train_samples)

# scaled_test_samples = scaler.fit_transform((test_samples).reshape(-1,1))

# predictions = model.predict(scaled_test_samples, batch_size=10, verbose=2)
# for i in predictions:
#     print(i)

# rounded_predictions = model.predict_classes(scaled_test_samples, batch_size=10, verbose=2)
# for i in rounded_predictions:
#     print(i)


# -> Confusion Matrix

