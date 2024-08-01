

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
        self.data_path = '../NetworkPackets.csv' # path for training data 
       
       
    def load(self):
        
        data = pandas.read_csv(self.data_path) 
        dataset = data.values 
       
        df_data = dataset[:, 0:11].astype(float)
        df_label = dataset[:, 11]

        return df_data,df_label

