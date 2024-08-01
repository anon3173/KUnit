

from sklearn import preprocessing
from sklearn import datasets
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
        #self.data_path = '/' # path for training data 
    
    def load(self):
        samples = datasets.load_iris()
        X = samples.data
        y = samples.target
        df = pandas.DataFrame(data=X)
        df.columns = samples.feature_names
        df['Target'] = y

        # prepare data
        X = df[df.columns[:-1]]
        y = df[df.columns[-1]]

        # hot encoding
        # encoder = LabelEncoder()
        # y1 = encoder.fit_transform(y)
        # # y = pd.get_dummies(y1).values
        y = to_categorical(y)
       
        df_data = pandas.DataFrame(X)
        df_label = pandas.DataFrame(y)
        return df_data,df_label
