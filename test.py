
import numpy as np
from neuralNetwork.layer import Layer
from neuralNetwork.activation import Sigmoid, Relu
from neuralNetwork.cost import RMSE
from neuralNetwork.network import FeedForwardNN

if __name__ == "__main__":
    
    trainX = np.random.randn(1000, 13) # number of samples, number of features
    trainY = np.random.randn(1000, 2) # number of samples, number of outputs


    # test
    layers = [
        Layer(10, Relu),
        Layer(5, Sigmoid),
        Layer(2, Sigmoid)
    ]

    myNN = FeedForwardNN( 13, layers=layers, costFunction=RMSE)

    myNN.compile()

    #myNN.fit(trainX, trainY, learningRate=0.001, numberOfEpoch=10)
    print myNN.predict( trainX )
    # 0.23 0.77