
import numpy as np
class CostFunction(object):
    def __init__(self):
        pass


class RMSE(CostFunction):
    @staticmethod
    def calc(y, yPredict):
        error = np.sum((y-yPredict)**2)
        #print "error", error.shape
        return error
        

    @staticmethod
    def derive(y, yPredict):
        #print "derivative", 2*(y-yPredict)
        return 2*(y-yPredict)


SAMPLE_DATA_COST_Y = np.array([[0.14664598, 0.10012058],
       [0.58958759, 0.698994  ],
       [0.61672428, 0.49692433],
       [0.91638048, 0.01091912],
       [0.15890042, 0.24084781]])

SAMPLE_DATA_COST_Y_PREDICT = np.array([[0.93921682, 0.7959142 ],
       [0.7862682 , 0.88657393],
       [0.71274709, 0.05922099],
       [0.72522229, 0.82836381],
       [0.37858954, 0.43062208]])

result = RMSE.calc(SAMPLE_DATA_COST_Y, SAMPLE_DATA_COST_Y_PREDICT)
#print result
