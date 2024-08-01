

from sklearn import preprocessing
from sklearn.datasets import load_iris, make_classification
from tensorflow.keras.utils import to_categorical
from keras.utils import np_utils 
import pandas
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler
import tensorflow 
import numpy as np
import seaborn as sns
from sklearn.datasets import make_regression

class DataPreprocess():
    
    def __init__(self):
        super(DataPreprocess, self).__init__()
        # self.data_path = '' # path for training data 
   
    def load(self):
        
        # dataset = pandas.read_csv(self.data_path,header=None)
      
        X_train, y_train = make_classification(n_samples=100, n_features= 14,n_informative = 10,n_classes=11, n_clusters_per_class=1,)
        y_train = to_categorical(y_train)
        df_data = X_train
        df_label = y_train
        return df_data,df_label

