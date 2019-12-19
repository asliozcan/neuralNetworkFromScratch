
import numpy as np
from neuralNetwork.layer import Layer
from neuralNetwork.activation import Sigmoid, Relu
from neuralNetwork.cost import RMSE
from neuralNetwork.network import FeedForwardNN
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder



if __name__ == "__main__":
    
    # trainX = np.random.randn(1000, 13) # number of samples, number of features
    # trainY = np.random.randn(1000, 2) # number of samples, number of outputs

    df = pd.read_csv('~/Documents/git.basestech.com/neural-network-from-scratch/neuralNetwork/dataset/heart.csv')
    #print df.shape

    y= df.target
    X = df.iloc[: , :-1].values
    #print(np.shape(X))
    y = df.iloc[: , 13:14].values
    #print(np.shape(y))

    imp = Imputer(missing_values=np.nan, strategy='mean')
    imputer = imp.fit(X[:, :13])
    X[:, 0:13] = imputer.transform(X[:, 0:13])

    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(y)
    y= enc.transform(y).toarray()
    X= X / X.max(axis=0)

    trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.25, random_state=42)

    # test
    layers = [
        Layer(10, Relu),
        Layer(5, Sigmoid),
        Layer(2, Sigmoid)
    ]

    layers2 = [
        Layer(20, Sigmoid),
        Layer(5, Sigmoid),
        Layer(2, Sigmoid)
    ]

    # myNN = FeedForwardNN( 13, layers=layers, costFunction=RMSE)
    myNN2 = FeedForwardNN (13, layers=layers2, costFunction=RMSE)

    # myNN.compile()
    myNN2.compile()

    # myNN.fit(trainX, trainY, learningRate=0.001, numberOfEpoch=10)
    myNN2.fit(trainX, trainY, learningRate=0.001, numberOfEpoch=10)
    # print myNN.predict( trainX )
    # 0.23 0.77
    #print myNN2.predict( trainX)