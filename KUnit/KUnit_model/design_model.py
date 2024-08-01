
import tensorflow  
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense,Conv1D, MaxPooling1D, Flatten
class MyModel(tensorflow.keras.layers.Layer):
     
    def __init__(self):
        super(MyModel, self).__init__()
         
         
    def call(self):

        # Design model here
        model = Sequential()
        model.add(Dense(20, activation='relu', input_shape = (7,)))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(units = 1, activation='linear'))
        opti = Adam(learning_rate= 0.1)
        model.compile(optimizer = opti, loss='mean_squared_error', metrics=['mean_absolute_error'])
        return model
def main():
        dp = MyModel()
        dp.call()
   
   
if __name__ == "__main__":
    main()
    