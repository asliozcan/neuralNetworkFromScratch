
import numpy as np
class CostFunction(object):
    def __init__(self):
        pass


class RMSE(CostFunction):
    @staticmethod
    def calc(y, yPredict):
        """RMSE calculation
        
            :param y: type: -- Actual output layer values
            :param yPredict: type: -- Predicted output layer values
        """
        error = np.sum((y-yPredict)**2)
        print "error", error
        return error
        

    @staticmethod
    def derive(y, yPredict):
        """Cost Function's derivative
        
            :param y: type -- Actual output layer values
            :param yPredict: type -- Predicted output layer values
        """
        return 2*(y-yPredict)
