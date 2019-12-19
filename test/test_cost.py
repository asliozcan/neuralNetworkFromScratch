
import sys
import os
sys.path.append(os.path.abspath("../neuralNetwork"))
import pytest
import numpy as np
from tools import numericDerive, numericDerive2
from samples import SAMPLE_DATA_SIGMOID, SAMPLE_DATA_RELU, SAMPLE_DATA_RELU_RESULT, SAMPLE_DATA_SIGMOID_RESULT, SAMPLE_DATA_COST_Y, SAMPLE_DATA_COST_Y_PREDICT, COST_SAMPLE_RESULT

from neuralNetwork.cost import CostFunction, RMSE

def test_cost_calc():
    x = RMSE.calc(SAMPLE_DATA_COST_Y, SAMPLE_DATA_COST_Y_PREDICT)
    result = COST_SAMPLE_RESULT
    assert ( np.abs(x-result).sum() / np.abs(x).sum() ) < 0.00001

# def test_cost_derivatives():
#     for cost in [RMSE]:
#         y = np.random.rand(227,2)
#         yPredict = np.random.rand(227,2)
#         derivative =  cost.derive(y, yPredict)
#         numericDerivative = numericDerive2(cost.calc, y, yPredict)
#         assert ( np.abs(derivative-numericDerivative).sum() / np.abs(derivative).sum() ) < 0.001