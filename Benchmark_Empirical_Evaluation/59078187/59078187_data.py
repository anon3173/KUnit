

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
       
        x_train = np.linspace(0,10,1000)
        y_train = np.power(x_train,2.0)

        x_test = np.linspace(8,12,100)
        y_test = np.power(x_test,2.0)
        df_data = x_train
        df_label = y_train
        return df_data,df_label

