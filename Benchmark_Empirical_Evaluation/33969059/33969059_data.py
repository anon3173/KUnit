

from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical
from keras.utils import np_utils 
import pandas
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler
import tensorflow 
import numpy as np
import sys
from sklearn.datasets import make_regression

class DataPreprocess():
    
    def __init__(self):
        super(DataPreprocess, self).__init__()
        # self.data_path = '' # path for training data 
    def myrand(a, b) :
        return (b)*(np.random.random_sample()-0.5)+a

    def get_data(self,count, ws, xno, bounds=100, rweight=0.0) :
        mn = 1
        xt = np.random.rand(count, len(ws))
        xt = np.multiply(bounds, xt)
        yt = np.random.rand(count, 1)
        ws = np.array(ws, dtype=np.float)
        xno = np.array([float(xno) + rweight*self.myrand(-mn, mn) for x in xt], dtype=np.float)
        yt = np.dot(xt, ws)
        yt = np.add(yt, xno)

        return (xt, yt)  
        
    def load(self):
        NUM_TRAIN = 100000
        NUM_TEST = 10000
        INDIM = 3
        # df = pandas.read_csv(self.data_path,header=None)
        if 0 > 1 :
            EPOCHS = int(sys.argv[1])
            XNO = float(sys.argv[2])
            WS = [float(x) for x in sys.argv[3:]]
            mx = max([abs(x) for x in (WS+[XNO])])
            mn = min([abs(x) for x in (WS+[XNO])])
            mn = min(1, mn)
            WS = [float(x)/mx for x in WS]
            XNO = float(XNO)/mx
            INDIM = len(WS)
        else :
            INDIM = 3
            WS = [2.0, 1.0, 0.5]
            XNO = 2.2
            EPOCHS = 20

        X_test, y_test = self.get_data(10000, WS, XNO, 10000, rweight=0.4)
        X_train, y_train = self.get_data(100000, WS, XNO, 10000)
        df_data = X_train
        df_label = y_train
        return df_data,df_label


    