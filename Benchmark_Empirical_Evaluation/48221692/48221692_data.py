

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
        self.data_path = '../Linear_Data.csv' # path for training data 
       
       
    def load(self):
        
        df = pandas.read_csv(self.data_path,header=None)
        df = df.values
        df_data = df[:,0:1]
        df_label =  df[:,1]
       
        return df_data,df_label

