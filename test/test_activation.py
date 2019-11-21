
import sys
import os
sys.path.append(os.path.abspath("../neuralNetwork"))
import pytest
import numpy as np
from tools import numericDerive, numericDerive2
from samples import SAMPLE_DATA_SIGMOID, SAMPLE_DATA_RELU, SAMPLE_DATA_RELU_RESULT, SAMPLE_DATA_SIGMOID_RESULT

from neuralNetwork.activation import Sigmoid, Relu


def test_activation_derivatives():
    for act in [Sigmoid, Relu]:
        x = np.random.rand(5,5)
        dx =  act.derive(x) 
        dxNum = numericDerive( act.calc, x )
        assert ( np.abs(dx-dxNum).sum() ) < 0.00001

def test_sigmoid_calc():
    x = Sigmoid.calc(SAMPLE_DATA_SIGMOID)
    result = SAMPLE_DATA_SIGMOID_RESULT
    assert ( np.abs(x-result).sum() ) < 0.00001

def test_relu_calc():
    x = Relu.calc(SAMPLE_DATA_RELU)
    result = SAMPLE_DATA_RELU_RESULT
    assert ( np.abs(x-result).sum() ) < 0.00001