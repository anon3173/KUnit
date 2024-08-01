

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
        self.data_path = '../kc_house_data.csv' # path for training data 
    
    def load(self):
        df = pandas.read_csv(self.data_path) 
        df.isnull().sum()
        df = df.drop(['date', 'id', 'zipcode'], axis=1)
        from sklearn.model_selection import train_test_split
        X = df.drop(['price'], axis=1).values
        y = df['price'].values    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        df_data = pandas.DataFrame(X_train)
        df_label = pandas.DataFrame(y_train)
        return df_data,df_label
