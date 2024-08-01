

from sklearn import preprocessing
from sklearn.datasets import load_iris
from tensorflow.keras.utils import to_categorical
from keras.utils import np_utils 
import pandas
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler
import tensorflow 
import numpy as np
from sklearn.datasets import make_regression

class DataPreprocess():
    
    def __init__(self):
        super(DataPreprocess, self).__init__()
        self.data_path = '../dataset.csv' # path for training data 
       
       
    def load(self):
        
        dataset = pandas.read_csv(self.data_path,header=None)
        dataset= dataset.values
        X = dataset[:, 0:2]
        y = dataset[:, 2]

        df_data = X
        df_label = y
        return df_data,df_label


    