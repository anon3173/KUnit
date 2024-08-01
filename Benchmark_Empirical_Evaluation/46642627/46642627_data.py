

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
        # self.data_path = '' # path for training data 
       
       
    def load(self):
        
        # data = pandas.read_csv(self.data_path) 
        numpy.random.seed(7)

        # split into input (X) and output (Y) variables
        df_data = []
        df_label = []
        count = 0

        while count < 10000:
            count += 1
            df_data += [count / 10000]
            numpy.random.seed(count)
            #Y += [numpy.random.randint(1, 101) / 100]
            df_label += [(count + 1) / 100]
        
        return df_data,df_label


    