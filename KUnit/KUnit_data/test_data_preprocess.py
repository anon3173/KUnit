
import unittest
import pandas
import numpy as np 
import tensorflow 
from data_preprocess import DataPreprocess
from data_phase_2 import DataPhase2
from generate_mock_model import GenerateMockModel
from history import LossAccuracyHistory

        
class MyDataTestPhase1(tensorflow.test.TestCase):
    @classmethod
    def setUpClass(self):
        super(MyDataTestPhase1, self).setUpClass()
        self.data = DataPreprocess()
        self.my_data, self.my_label = self.data.load()
        self.my_data = pandas.DataFrame(self.my_data)
        self.data2 = DataPhase2()
        self.number_of_classes, self.problem_type, self.model_type = self.data2.labelConsistency()
        self.hist = LossAccuracyHistory()
        self.mock_model = GenerateMockModel(self.problem_type, self.model_type, self.my_data, self.my_label)
        self.model = self.mock_model.MockModel()
    
       
    def test1missingValue(self):
        my_data = pandas.DataFrame(self.my_data)
        check_data = my_data.apply(lambda x : pandas.api.types.is_string_dtype(x)).any()
        if check_data == False:
            assert (my_data.isnull().values.any() == False), 'Missing Value --> Use fillna()'
        elif check_data == True:
            assert (check_data == False), 'Encode categorical data or check for any string used to represent missing value in data --> Use LabelEncoder '


    def test2infiniteValue(self):
        my_data1 = pandas.DataFrame(self.my_data)
        check_data = my_data1.apply(lambda x : pandas.api.types.is_string_dtype(x)).any()
        if check_data == False:
            count = np.isinf(my_data1).values.sum()
            assert (count == 0), 'Data has Infinite Value'
        elif check_data == True:
            assert (check_data == False), 'Encode categorical data --> Use LabelEncoder'

    
    def test3labelCount(self):
        if self.problem_type == 1:
            pass
        elif self.problem_type == 2:
            num_label = self.my_label
            num_label = np.array(num_label).reshape(-1)
            distinct_label_count = len(pandas.unique(num_label))
            assert (distinct_label_count == self.number_of_classes), 'Number of labels not matching with problem definition'
    
    def test4missingLabel(self):
        try:
            label = pandas.DataFrame(self.my_label)
            assert (label.isnull().values.any() == False), 'Missing Label'
        except ValueError as msg:
             print(msg) 


    def test5dataNormalization(self):
        my_data1 = pandas.DataFrame(self.my_data)
        check_data = my_data1.apply(lambda x : pandas.api.types.is_string_dtype(x)).any()
        if check_data == False:
            my_data = pandas.DataFrame(self.my_data)
            m = np.array(my_data.std(axis=0))
            if self.problem_type == 1:
                n = m.astype(int)
                if len(n) == 1:
                    assert ((n==1) or (np.all(self.my_data > -2) and np.all(self.my_data < 2)) == True), 'Data is not normalized or scaled'
                elif len(n) > 1:
                    assert ((len(np.unique(n))==1) or (np.all(self.my_data > -2) and np.all(self.my_data < 2)) == True), 'Data is not normalized or scaled'
            elif self.problem_type == 2: 
                n = m.astype(int)
                assert ((len(np.unique(n))==1) or (np.all(self.my_data > -2) and np.all(self.my_data < 2)) == True), 'Data is not normalized or scaled'
       
        elif check_data == True:
            assert (check_data == False), 'Encode categorical data --> Use LabelEncoder'
        
        
    def test6labelEncoding(self):
        l = pandas.DataFrame(self.my_label)
        check_label = l.apply(lambda x : pandas.api.types.is_string_dtype(x)).any()
        assert check_label==False ,'Encode the labels --> Use LabelEncoder'


    def test7classImbalance(self):
        my_label = pandas.DataFrame(self.my_label)
        if self.problem_type == 2: 
            unique_target = np.unique(my_label)
            k = len(unique_target)
            n = len(self.my_data)
            ci1 = []
            for target in unique_target:
                ci = (my_label.values == target).sum()
                ci1.append(ci)
            max_samples = np.max(ci1)
            min_samples = np.min(ci1)
            total = np.sum(ci1)
            imb = (max_samples - min_samples)/(total - k)
            assert (imb ==0), 'Class imbalance --> Balance the classes'
    
  
    def test0modelLearning(self):
        my_data1 = pandas.DataFrame(self.my_data)
        check_data = my_data1.apply(lambda x : pandas.api.types.is_string_dtype(x)).any()
        l = pandas.DataFrame(self.my_label)
        check_label = l.apply(lambda x : pandas.api.types.is_string_dtype(x)).any()
        if check_data == False and check_label == False:
            c = 0
            if self.problem_type == 1:
                if self.model_type == 'cnn':
                    self.my_data = self.my_data.reshape(-1, self.my_data.shape[1], 1)
                    self.model.fit(self.my_data,self.my_label,epochs=10,verbose=1,callbacks=[self.hist])
                elif self.model_type == 'dnn':
                    self.rows, self.cols = self.my_data.shape
                    self.model.fit(self.my_data,self.my_label,epochs=10,verbose=1,callbacks=[self.hist]) 
                if (self.hist.mae[0] > self.hist.mae[len(self.hist.losses)-1]): 
                        c =c+1
                self.model.summary()
                assert (c==1), 'Basic Model is not Learning'
            elif self.problem_type == 2:
                self.model.fit(self.my_data,self.my_label,epochs=5,verbose=1,callbacks=[self.hist]) 
                if ((self.hist.losses[0] > self.hist.losses[len(self.hist.losses)-1]) and (self.hist.accuracy[0] < self.hist.accuracy[len(self.hist.losses)-1])): 
                        pass
                else:
                        c =c+1
                self.model.summary()
                assert (c==0), 'Basic Model is not Learning'
        elif check_data == True or check_label == True:
            assert (check_data == False and check_label == False), 'Encode categorical data --> Use LabelEncoder'


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
    with open('../test_data_results.txt', 'a+') as f:
        # Create a test suite
        suite = unittest.TestLoader().loadTestsFromTestCase(MyDataTestPhase1)

        # Create a test runner and set the result output to the file
        runner = unittest.TextTestRunner(stream=f, resultclass=FileTestRunner)

        # Run the tests
        result = runner.run(suite)
        
       