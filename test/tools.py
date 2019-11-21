
import numpy as np


def numericDerive(func, x, h=np.finfo(np.float32).eps ):
    return (func(x+h) - func(x)) / h


def numericDerive2 (func, x, y, h=np.finfo(np.float32).eps ):
        print "numericDerivative", (func(x+h, y)-func(x, y)) / h
        return (func(x+h, y)-func(x, y)) / h