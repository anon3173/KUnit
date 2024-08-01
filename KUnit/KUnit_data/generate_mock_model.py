

from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
import tensorflow
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Activation, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam

class GenerateMockModel():
    def __init__(self,problem_type,model_type,x,y):
        super(GenerateMockModel, self).__init__()
        self.problem_type = problem_type
        self.model_type = model_type
        self.x = x
        self.y = y
        self.rows, self.cols = self.x.shape
       
        self.distinct_labels = np.unique(self.y, return_counts=False)
       
    def MockModel(self):
        if self.model_type == 'dnn':
            np.random.seed(42)
            tensorflow.random.set_seed(42)
            opt = Adam()
            if isinstance(self.y, list) == False:
                lo = 'sparse_categorical_crossentropy'
            elif isinstance(self.y[0], list):
                lo = 'categorical_crossentropy'
            model = Sequential()
            model.add(Dense(self.cols,input_shape=(self.cols,) , activation = 'relu'))
            
            if len(self.distinct_labels) ==2 and self.problem_type == 2:
              
                model.add(Dense(len(self.distinct_labels)-1, activation = 'sigmoid'))
                model.compile(loss = 'binary_crossentropy' , optimizer = opt , metrics = ['accuracy'] )
            elif len(self.distinct_labels) > 2 and self.problem_type == 2:
                model.add(Dense(len(self.distinct_labels), activation = 'softmax'))
                
                model.compile(loss = lo , optimizer = opt , metrics = ['accuracy'] )
                
            elif self.problem_type == 1:
                model.add(Dense(1))
                model.compile(loss= 'mean_squared_error', metrics=['mean_absolute_error'] , optimizer = opt )
        
        elif self.model_type == 'cnn':
            np.random.seed(42)
            tensorflow.random.set_seed(42)
            opt = Adam()
            model = Sequential()
            model.add(Conv1D(32,input_shape=(self.cols,1) , kernel_size=3, strides=1, padding='same', activation = 'relu'))
            model.add(MaxPooling1D(strides=2))
            model.add(Flatten())
            if len(self.distinct_labels) ==2 and self.problem_type == 2:
                model.add(Dense(len(self.distinct_labels)-1, activation = 'sigmoid'))
                model.compile(loss = 'binary_crossentropy' , optimizer = opt , metrics = ['accuracy'] )
            elif len(self.distinct_labels) > 2 and self.problem_type == 2:
                model.add(Dense(len(self.distinct_labels), activation = 'softmax'))
                model.compile(loss = 'sparse_categorical_crossentropy' , optimizer = opt , metrics = ['accuracy'] )
            elif self.problem_type == 1:
                model.add(Dense(1))
                model.compile(loss= 'mean_squared_error', metrics=['mean_absolute_error'] , optimizer = opt )


        return model
        

   