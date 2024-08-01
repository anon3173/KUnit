from keras.losses import CategoricalCrossentropy
import tensorflow as tf
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=1000, n_features=5, n_classes=5, n_informative=4, n_redundant=0, random_state=42)
sc = StandardScaler()
X = sc.fit_transform(X)

model = tf.keras.Sequential([
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dense(16, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(8, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(X, y, epochs=20, batch_size=24)
