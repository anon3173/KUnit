

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
        self.data_path = '../cars.csv' # path for training data 
       
       
    def load(self):
        
        df = pandas.read_csv(self.data_path,header=None)
        dataset = dataset.values
        x = dataset[:, 0:5]
        y = dataset[:, 5]
        df_label = np.reshape(y, (-1, 1))
        scaler = MinMaxScaler()
        df_data = scaler.fit_transform(x)
        return df_data,df_label

