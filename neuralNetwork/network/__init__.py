"""



"""
import numpy as np


class FeedForwardNN(object):
    def __init__(self, inputSize, layers=[], costFunction=None):
        """Feed Forward Neural Network Object     
        
        Arguments:
            object {[type]} -- 
            inputSize {int} -- 
            layers {Layer[]} -- Array of layers which are included in neural network
            costFunction {CostFunction} -- Cost function of neural network
        """
        self.inputSize = inputSize
        self.layers = layers
        self.costFunction = costFunction
        self._weights = []


    def compile(self):
        """Compile the model and initiliaze weights
        """
        self._weights.append(
            np.random.randn(self.inputSize, self.layers[0].numberOfNeurons )
        )
        for i in range(1,len(self.layers)):
            self._weights.append(
                np.random.randn( self.layers[i-1].numberOfNeurons, self.layers[i].numberOfNeurons )
            )
        

    def fit(self, trainX, trainY, learningRate=0.001, numberOfEpoch=10):
        """[summary]
        
        Arguments:
            trainX {[type]} -- [description]
            trainY {[type]} -- [description]
        
        Keyword Arguments:
            learningRate {float} -- [description] (default: {0.001})
            numberOfEpoch {int} -- [description] (default: {10})
        """
        pass

    def predict(self, x):
        """[summary]
        
        Arguments:
            input {[type]} -- [description]
        """
        r = self._forward(x)
        return r 


    def save(self, filename):
        """[summary]
        
        Arguments:
            filename {[type]} -- [description]
        """
    
    def laod(self, filename):
        """[summary]
        
        Arguments:
            filename {[type]} -- [description]
        """

    def _forward(self, x):
        print self.layers
        for i in range(0, len(self.layers)):
            print i, x.shape
            x = self.layers[i].activation.calc( np.dot( x , self._weights[i] ) ) 
        return x

    def _backprop(self, y):
        pass


