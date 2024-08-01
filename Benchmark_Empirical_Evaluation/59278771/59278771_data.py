

from sklearn import preprocessing
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
        self.data_path = '../iris.data' # path for training data 
       
       
    def load(self):
        
        dataset = pandas.read_csv(self.data_path,header=None)
        dataset = dataset.values 
        df_data = dataset[:,0:4].astype(float) 
        Y = dataset[:,4] 

        # encode class values as integers 
        encoder = LabelEncoder() 
        encoder.fit(Y) 
        df_label = encoder.transform(Y) 

        return df_data,df_label

