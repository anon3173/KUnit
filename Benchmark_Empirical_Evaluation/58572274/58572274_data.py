

from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical
from keras.utils import np_utils 
import pandas
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
        
        data = np.array([[0., 0., 1.], [0., 1., 0.], [1., 0., 0.], [1., 1., 1.]])
        train = data[:, :-1]  # Taking The same and All data for training
        test = data[:, :-1]

        train_l = data[:, -1]
        test_l = data[:, -1]

        train_label = []
        test_label = []

        for i in train_l:
            train_label.append([i])
        for i in test_l:
            test_label.append([i])  # Just made Labels Single element...

        train_label = np.array(train_label)
        test_label = np.array(test_label)  # Numpy Conversion
        df_data = train
        df_label = train_label
        return df_data,df_label


    