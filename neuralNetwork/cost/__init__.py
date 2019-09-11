
import numpy as np
class CostFunction(object):
    def __init__(self):
        pass


class RMSE(CostFunction):
    @staticmethod
    def calc(y, yPredict):
        error = np.sum((y-yPredict)**2)
        return error
        

    @staticmethod
    def derive(y, yPredict):
        return 2*(y-yPredict)
