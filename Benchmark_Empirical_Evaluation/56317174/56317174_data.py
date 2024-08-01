

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
        self.data_path = '../weatherAUS.csv' # path for training data 
       
       
    def load(self):
        
        df = pandas.read_csv(self.data_path,header=None)
        csv_data = df.replace("NA", 0.0, regex=True)


        # Input/output columns scaled to 0<=n<=1
        x = csv_data.loc[:, ['MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed']]
        x = x.fillna(0)
        lb = LabelEncoder
        y = lb.fit_transform(csv_data['RainTomorrow'])
        print(x)
        print(y)
        scaler_x = MinMaxScaler(feature_range=(-1,1))
        x = scaler_x.fit_transform(x)
        df_data = x
        df_label = y
        return df_data,df_label


    