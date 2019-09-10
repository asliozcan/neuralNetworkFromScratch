

import numpy as np


class Activation(object):
    def __init__(self):
        pass

class Softmax(Activation):
    @staticmethod
    def calc(x):
        pass

    @staticmethod
    def derive(x):
        pass

class Sigmoid(Activation):
    @staticmethod
    def calc(x):
        sigm = 1 / (1 + np.exp(-x))
        return sigm

    @staticmethod
    def derive(x):
        sigm = Sigmoid.calc(x)
        return sigm * (1 - sigm)

class Relu(Activation):
    @staticmethod
    def calc(x):
        return np.maximum(0, x)

    @staticmethod
    def derive(x):
        x[x<=0] = 0
        x[x>0] = 1
        return x
