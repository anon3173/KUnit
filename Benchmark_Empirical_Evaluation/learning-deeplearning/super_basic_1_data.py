

from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical
from keras.utils import np_utils 
import pandas
import sys
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler
import tensorflow 
import numpy as np
import random
import seaborn as sns
from sklearn.datasets import make_regression, make_classification


class DataPreprocess():
    
    def __init__(self):
        super(DataPreprocess, self).__init__()
        # self.data_path = '' # path for training data 
    
    def load(self):
        
        train_labels = []
        train_samples = []

        for i in range(50):
            random_younger = random.randint(13, 64)
            train_samples.append(random_younger)
            train_labels.append(1)

            random_older = random.randint(65, 100)
            train_samples.append(random_older)
            train_labels.append(0)

        for i in range(1000):
            random_younger = random.randint(13, 64)
            train_samples.append(random_younger)
            train_labels.append(0)

            random_older = random.randint(65, 100)
            train_samples.append(random_older)
            train_labels.append(1)

        # -> Preprocessing

        train_labels = np.array(train_labels)
        train_samples = np.array(train_samples)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_train_samples = scaler.fit_transform((train_samples).reshape(-1, 1))
        df_data = pandas.DataFrame(scaled_train_samples)
        df_label = pandas.DataFrame(train_labels)
        return df_data,df_label
 