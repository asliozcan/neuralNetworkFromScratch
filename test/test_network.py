
import sys
import os
sys.path.append(os.path.abspath("../neuralNetwork"))
import pytest
import numpy as np
from tools import numericDerive, numericDerive2
from samples import SAMPLE_DATA_SIGMOID, SAMPLE_DATA_RELU, SAMPLE_DATA_RELU_RESULT, SAMPLE_DATA_SIGMOID_RESULT, SAMPLE_DATA_COST_Y, SAMPLE_DATA_COST_Y_PREDICT, COST_SAMPLE_RESULT

from neuralNetwork.layer import Layer
from neuralNetwork.activation import Sigmoid, Relu
from neuralNetwork.cost import RMSE
from neuralNetwork.network import FeedForwardNN

def test_network():
    trainX = np.random.randn(1000, 13) # number of samples, number of features
    trainY = np.random.randn(1000, 2) # number of samples, number of outputs
    layers1 = [
        Layer(10, Relu),
        Layer(5, Sigmoid),
        Layer(2, Sigmoid)
    ]

    layers2 = [
        Layer(20, Sigmoid),
        Layer(5, Sigmoid),
        Layer(2, Sigmoid)
    ]
    layers =[layers1, layers2]
    for layer in layers:
        myNN = FeedForwardNN(13, layer, RMSE)
        myNN.compile()
        myNN.fit(trainX, trainY, learningRate=0.001, numberOfEpoch=10)
        myNN.predict(trainX)