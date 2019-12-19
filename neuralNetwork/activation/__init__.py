

import numpy as np


class Activation(object):
    def __init__(self):
        pass

class Softmax(Activation):
    @staticmethod
    def calc(x):
        """Activation function calculation
        
    
            :param x: type: -- Layer values
        """
        pass

    @staticmethod
    def derive(x):
        pass

class Sigmoid(Activation):
    @staticmethod
    def calc(x):
        """Sigmoid activation function calculation
        
            :param x: type: -- Layer values
        """
        sigm = 1 / (1 + np.exp(-x))
        return sigm

    @staticmethod
    def derive(x):
        """Derivative of Sigmoid function
        
            :param x: type: -- Layer values
        """
        sigm = Sigmoid.calc(x)
        return sigm * (1 - sigm)

class Relu(Activation):
    @staticmethod
    def calc(x):
        """Relu activation function calculation
        
       
            :param x: type -- Layer values
        """
        return np.maximum(0, x)

    @staticmethod
    def derive(x):
        """Derivative of Relu function
        
            :param x: type -- Layer values
        """
        x[x<=0] = 0
        x[x>0] = 1
        return x
