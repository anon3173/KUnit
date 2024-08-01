

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
        n_samples = 200
        n_feats = 2

        cls0 = np.random.uniform(low=0.2, high=0.4, size=(n_samples,n_feats))
        cls1 = np.random.uniform(low=0.5, high=0.7, size=(n_samples,n_feats))
        x_train = np.concatenate((cls0, cls1))
        y_train = np.concatenate((np.zeros((n_samples,)), np.ones((n_samples,))))

        # shuffle data because all negatives (i.e. class "0") are first
        # and then all positives (i.e. class "1")
        indices = np.arange(x_train.shape[0])
        np.random.shuffle(indices)
        df_data = x_train[indices]
        df_label = y_train[indices]

        return df_data,df_label


    