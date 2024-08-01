

from sklearn import preprocessing
from keras.utils import np_utils 
import pandas
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler
import tensorflow 
import numpy 

class DataPreprocess():
    
    def __init__(self):
        super(DataPreprocess, self).__init__()
        self.data_path = '../sorted output.csv' # path for training data 
       
       
    def load(self):

        data = pandas.read_csv(self.data_path) 
        data = data.values 
        dataset = dataset.values 
        df_data = dataset[:,0:3]
        df_label = dataset[:,3]
      

        return df_data,df_label

