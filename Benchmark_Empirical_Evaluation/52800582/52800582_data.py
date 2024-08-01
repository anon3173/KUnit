

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
        # self.data_path = '' # path for training data 
       
       
    def load(self):
        
        # dataset = pandas.read_csv(self.data_path,header=None)

        df_data = np.arange(-100, 100, 0.5)
        df_label = df_data**4
	df_data = df_data.reshape(400,1)
        # encode class values as integers 
        
        return df_data,df_label

