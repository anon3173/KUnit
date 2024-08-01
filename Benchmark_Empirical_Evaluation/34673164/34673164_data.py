

from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical
from keras.utils import np_utils 
import pandas
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler
import tensorflow 
import numpy 

class DataPreprocess():
    
    def __init__(self):
        super(DataPreprocess, self).__init__()
        self.data_path = '../dataset.csv' # path for training data 
       
       
    def load(self):

        data = pandas.read_csv(self.data_path) 
        data = data.values 
        df_data = data[:,0:16].astype(float) 
        df_label = data[:,16]

        # encode class values as integers 
        encoder = LabelEncoder() 
        encoder.fit(df_label) 
        encoded_Y = encoder.transform(df_label) 

        # convert integers to dummy variables (i.e. one hot encoded) 
        df_label = np_utils.to_categorical(encoded_Y) 


        return df_data,df_label

