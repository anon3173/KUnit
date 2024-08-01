

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
        train_samples = []
        train_labels = []

        for i in range(50):
            # The ~5% of younger individuals who did experience side effects
            random_younger = random.randint(13,64)
            train_samples.append(random_younger)
            train_labels.append(1)

            # The ~5% of older individuals who did not experience side effects
            random_older = random.randint(65,103)
            train_samples.append(random_older)
            train_labels.append(0)

        for i in range(1000):
            # The ~95% of younger individuals who did not experience side effects
            random_younger = random.randint(13,64)
            train_samples.append(random_younger)
            train_labels.append(0)

            # The ~95% of older individuals who did experience side effects
            random_older = random.randint(65,103)
            train_samples.append(random_older)
            train_labels.append(1)
        
        # convert both lists into numpy arrays(due to the fit() function expects)
        train_labels = np.array(train_labels)
        train_samples = np.array(train_samples)
        # shuffle the arrays to remove any order that was imposed on the data during the creation process
        train_labels, train_samples = random.shuffle(train_labels, train_samples)

        # scale the data down to a range from 0 to 1
        # by using scikit-learn’s MinMaxScaler class to scale all of the data down 
        # from a scale ranging from 13 to 103 to be on a scale from 0 to 1
        scaler = MinMaxScaler(feature_range=(0,1))
        # reshaping the data as a technical requirement
        # since the fit_transform() function doesn’t accept 1D data by default
        scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1,1))
        df_data = pandas.DataFrame(scaled_train_samples)
        df_label = pandas.DataFrame(train_labels)
        return df_data,df_label
