
import unittest
from sklearn.preprocessing import OneHotEncoder
from design_model import MyModel
from interface_test_phase_2 import InterfacePhase2
import tensorflow 
from generate_mock_data import GenerateMockData
from loss_accuracy_history import LossAccuracyHistory
        
class MyModelTest2(tensorflow.test.TestCase):
    @classmethod
    def setUpClass(self):
        super(MyModelTest2, self).setUpClass()
        self.model = MyModel()
        self.my_model = self.model.call()
        self.mock = GenerateMockData()
        self.X1,self.y,self.problem_type = self.mock.MockData()
        self.hist = LossAccuracyHistory()
        opt = self.my_model.loss
        if 'dense' in self.my_model.layers[0].name or 'dense' in self.my_model.layers[1].name:
            pass
        elif 'conv1d' in self.my_model.layers[0].name or 'conv1d' in self.my_model.layers[1].name :
            self.X1 = self.X1.reshape(-1, self.X1.shape[1], 1)
        if opt == 'sparse_categorical_crossentropy':
            self.history = self.my_model.fit(self.X1,self.y,batch_size=10,epochs=30,verbose=0,callbacks=[self.hist])
        elif opt == 'categorical_crossentropy':
            
            y1 = self.y.reshape(-1, 1)
            encoder = OneHotEncoder(sparse=False)
            y = encoder.fit_transform(y1)
            self.history = self.my_model.fit(self.X1,y,batch_size=10,epochs=30,verbose=1,callbacks=[self.hist])
        elif opt == 'binary_crossentropy':
           
            self.history = self.my_model.fit(self.X1,self.y,batch_size=10,epochs=30,verbose=1,callbacks=[self.hist])
        else:
            self.history = self.my_model.fit(self.X1,self.y,batch_size=10,epochs=30,verbose=1,callbacks=[self.hist])
      
        self.my_model.summary()

    def testSlowConvergence(self):
       
        if self.problem_type == 1:
            j = len(self.hist.met) - 1
            met = (self.hist.losses[0] - self.hist.losses[j]) > 1
            assert met == True, 'Slow Convergence --> Change Optimizer/Learning Rate'
    
        if self.problem_type == 2:
            c=0
            j = len(self.hist.met) - 1
            for i in range(0,25,5):
                if ((self.hist.met[i+5] - self.hist.met[i])/self.hist.met[i])*100 >=0:
                    pass
                else:
                    c =c+1
            assert (c == 0) , 'Slow Convergence --> Change Optimizer/Learning Rate'
    
    def testOscillatingLoss(self):
        c = 0
        for i in range(0,25,5):
                if ((self.hist.losses[i] - self.hist.losses[i+5]) > 0): 
                    pass
                else:
                    c =c+1
        assert (c == 0) , 'Oscillating Loss --> Change Learning Rate'
    
    def testAccuracyNotChanging(self):
        if self.problem_type == 2:
            i = len(self.hist.met) - 1
            assert self.hist.met[i] > self.hist.met[0]  , 'Stuck Accuracy --> Change Optimizer/Learning Rate'


class FileTestRunner(unittest.TextTestResult):
    def addSuccess(self, test):
        super().addSuccess(test)
        print('Test passed',test.id)
        self.stream.write(f"{test.id()}: Passed\n")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        print(f"Error in test {test.id()}: {err}")
        self.stream.write(f"{test.id()}: Failed\n")

if __name__ == '__main__':
    # Specify the file to write results
    with open('../test_model_results.txt', 'a+') as f:
        # Create a test suite
        suite = unittest.TestLoader().loadTestsFromTestCase(MyModelTest2)

        # Create a test runner and set the result output to the file
        runner = unittest.TextTestRunner(stream=f, resultclass=FileTestRunner)

        # Run the tests
        result = runner.run(suite)



