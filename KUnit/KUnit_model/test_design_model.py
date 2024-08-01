
from design_model import MyModel
from specification import Spec
import tensorflow 
import unittest
        
class MyModelTest(tensorflow.test.TestCase):
    @classmethod
    
    def setUpClass(self):
        super(MyModelTest, self).setUpClass()
        self.model = MyModel()
        self.my_model = self.model.call()
        self.spec = Spec()
        self.problem_type, self.classification_type, self.number_of_classes = self.spec.specification()
        self.features = self.spec.features()
    
   
    def test1inputShape(self):
        input_shape = (self.my_model.layers[0].__getattribute__('input_shape'))[1]
        try:
              assert input_shape == self.features,'Mismatch in input shape'
        except ValueError as msg:
             print(msg) 

    def test2outputShape(self):
           
        try:
            if self.my_model.layers[len(self.my_model.layers)-1].name.find('activation')!=-1:
                if self.my_model.layers[len(self.my_model.layers)-2].name.find('dense')!=-1:
                    unit = int(self.my_model.layers[len(self.my_model.layers)-2].__getattribute__('units'))
                    if self.problem_type == 1:
                        assert unit == 1,'Mismatch in output shape --> For regression use units = 1'
                    if self.problem_type == 2 and self.number_of_classes == 2:
                        assert unit == 1,'Mismatch in output shape --> For binary classification use units = 1'
                    elif self.problem_type == 2 and self.number_of_classes > 2:
                        assert unit == self.number_of_classes, 'Mismatch in output shape --> number of units should be same as number of classes'

            elif self.my_model.layers[len(self.my_model.layers)-1].name.find('dense')!=-1:
                    unit = int(self.my_model.layers[len(self.my_model.layers)-1].__getattribute__('units'))
                    if self.problem_type == 1:
                        assert unit == 1,'Mismatch in output shape --> For regression use units = 1'
                    if self.problem_type == 2 and self.number_of_classes == 2:
                        assert unit == 1,'Mismatch in output shape --> For binary classification use units = 1'
                    elif self.problem_type == 2 and self.number_of_classes > 2:
                        assert unit == self.number_of_classes, 'Mismatch in output shape --> number of units should be same as number of classes'
        except ValueError as msg:
             print(msg)   

    def test3outputLayerActivation(self):
           
        try:
            act = str(self.my_model.layers[len(self.my_model.layers)-1].__getattribute__('activation')).split()[1]
            if self.problem_type == 1:
                assert act == 'linear','Incorrect output layer activation --> Use linear activation'
            elif self.problem_type == 2 and self.classification_type == 1:
                assert act == 'sigmoid','Incorrect output layer activation --> Use sigmoid activation'
            elif self.problem_type == 2 and self.classification_type == 2:
                assert act == 'softmax','Incorrect output layer activation --> Use softmax activation'
        except ValueError as msg:
             print(msg)      
    
    def test4lossFunction(self):
       
        opt = self.my_model.loss
        try:
            act = str(self.my_model.layers[len(self.my_model.layers)-1].__getattribute__('activation')).split()[1]
            if self.problem_type == 1:
                mesg =[]
                if opt == 'mean_squared_error' or  opt=='mean_absolute_error' or opt == 'mse' or opt == 'mae':
                    pass 
                else:
                    mesg.append('Incorrect loss --> Use mean squared error or mean absolute error')
                assert len(mesg) == 0 ,mesg
            elif self.problem_type == 2 and self.classification_type == 1 :
                assert opt == 'binary_crossentropy','Incorrect loss --> Use binary crossentropy'
            elif self.problem_type == 2 and self.classification_type == 2 :
                assert opt == 'categorical_crossentropy','Incorrect loss --> Use categorical crossentropy'
        except ValueError as msg:
             print(msg)
    
    def test5Metric(self):
        try:
            if self.problem_type == 1:
                mesg =[]
                if self.my_model.compiled_metrics._metrics == None:
                    mesg.append('Add metrics --> Use mean squared error or mean absolute error')

                elif len(self.my_model.compiled_metrics._metrics) == 1:
                    met = self.my_model.compiled_metrics._metrics[0]
                    if met == 'mean_squared_error' or  met=='mean_absolute_error' or met == 'mse' or met == 'mae':
                        pass 
                    else:
                        mesg.append('Incorrect metrics --> Use mean squared error or mean absolute error')
                assert len(mesg) == 0 ,mesg[0]
               
            elif self.problem_type == 2 :
                mesg =[]
                if self.my_model.compiled_metrics._metrics == None:
                    mesg.append('Add metrics --> Use accuracy')
                elif len(self.my_model.compiled_metrics._metrics) == 1:
                    met = self.my_model.compiled_metrics._metrics[0]
                    if met == 'accuracy' :
                        pass 
                    else:
                        mesg.append('Incorrect metrics --> Use accuracy')
                assert len(mesg) == 0 ,mesg[0]
            
        except ValueError as msg:
             print(msg)
        

    # def test6learningRate(self):
           
    #         lr = self.my_model.optimizer.learning_rate 
    #         learn_rate = lr.numpy()
    #         try:
    #             msg =[]
    #             if learn_rate >= 0.01:
    #                 msg.append('Learning Rate too high: Decrease the Learning Rate')
            
    #             elif learn_rate < 0.0001:
    #                 msg.append('Learning Rate too low: Increase the Learning Rate')
                   
    #             assert len(msg) == 0 ,msg[0]
    #         except ValueError as msg:
    #             print(msg)

    def test7nonLinearity(self):
       
        names = []
        for layer in self.my_model.layers:
            names.append(layer.name)
        i=0
        for layer in self.my_model.layers:
            j = 0
            act = -1
            
            if i < len(names)-1:
                if names[i].find('dense')!=-1 or names[i].find('conv1d')!=-1  :
                    
                    if str(layer.activation).find('linear') !=-1:
                        
                        if names[i+1].find('activation')!=-1:
                            
                            act = 1
                        elif names[i+1].find('activation')==-1:
                            act = 0
                    assert (act == 1 or act == -1), 'Missing non-linear activation function in hidden layers'
                i = i+1
  
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
        suite = unittest.TestLoader().loadTestsFromTestCase(MyModelTest)

        # Create a test runner and set the result output to the file
        runner = unittest.TextTestRunner(stream=f, resultclass=FileTestRunner)

        # Run the tests
        result = runner.run(suite)

