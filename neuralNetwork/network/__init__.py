"""



"""
import numpy as np


class FeedForwardNN(object):
    def __init__(self, inputSize, layers=[], costFunction=None):
        """Feed Forward Neural Network Object     
        
       
            :param object: type: -- 
            :param inputSize: int: -- Size of input layer
            :param layers: :Layer[]: -- Array of layers which are included in neural network
            :param costFunction: CostFunction: -- Cost function of neural network
        """
        self.inputSize = inputSize
        self.layers = layers
        self.costFunction = costFunction
        self._weights = []
        self.d_weights = []
        self.layerCache=[]


    def compile(self):
        """Compile the model and initiliaze weights
        """
        self._weights.append(
            np.random.randn(self.inputSize, self.layers[0].numberOfNeurons )
        )
        for i in range(1,len(self.layers)):
            self._weights.append(
                np.random.randn( self.layers[i-1].numberOfNeurons, self.layers[i].numberOfNeurons )
            )
        

    def fit(self, trainX, trainY, learningRate=0.001, numberOfEpoch=10):
        """Fit the model and make prediction
        
            :param trainX: type: -- Training feature set
            :param trainY: type: -- Training target set
        
            :param learningRate: float: -- The step size at each iteration while moving toward a minimum of a loss function (default: {0.001})
            :param numberOfEpoch: int: -- Number of times all of the training vectors are used once to update weights  (default: {10})
        """
        for i in range(0, numberOfEpoch):
            yPredict = self._forward(trainX)
            print "shapes", yPredict.shape, trainY.shape
            self._backprop(trainY, yPredict)
            #self.show()
            #print len(self.d_weights)
            for j in range(0, len(self._weights)):
                shift = self.d_weights[j] * learningRate
                self._weights[j] = self._weights[j] + shift
            print i, ". epoch is ended"
            acc = trainY - yPredict
            print "acc", acc
            self.costFunction.calc(trainY, yPredict)
        


    def predict(self, x):
        """Predict an output layer values
        
            :param x: type: -- Training feature set
        """
        r = self._forward(x)
        return r 


    def save(self, filename):
        """Save the file

            :param filename: str: -- filename
        """
    
    def laod(self, filename):
        """Load the file
        
            :param filename: str: -- Filename
        """

    def _forward(self, x):
        """Forward propagation for NN
        
            :param x: type: -- Training feature set
        """
        self.layerCache=[]
        for i in range(0, len(self.layers)):
            #raw_input("wait forw")
            self.layerCache.append(x)
            x = self.layers[i].activation.calc( np.dot( x , self._weights[i] ) )
        return x

    def _backprop(self, y, yPredict):
        """Back propagation for NN
        
            :param y: type: -- Actual output layer values
            :param yPredict: type: -- Predicted output layer values
        """
        self.d_weights = []
        parameter=(self.costFunction.derive(y, yPredict)*self.layers[len(self.layers)-1].activation.derive(yPredict))
        #print parameter.shape
        # print "parameters", parameter, self.show()
        #print "###############", len(self._weights), len(self.layerCache), "#########"
        for k in range(len(self._weights), 0, -1):
            if k==len(self._weights): 
                dweight = np.dot(self.layerCache[k-1].transpose(), parameter)
                #print "dweight", k, dweight, dweight.shape
                self.d_weights.append(dweight)
            else:
                parameter = np.dot(parameter, self._weights[k].transpose())*self.layers[k].activation.derive(self.layerCache[k])
                dweight = np.dot(self.layerCache[k-1].transpose(), parameter)
                #print "dweight", k, dweight, dweight.shape
                self.d_weights.append(dweight)
        self.d_weights.reverse()
        #print [x for x in self.d_weights]
        return self.d_weights


    def show(self):
        """Show the NN objects attributes
        """
        print "##########"
        print "W: ", [x.shape for x in self._weights]
        print "derivatives: ", [x.shape for x in self.d_weights]
        print "LC: ", [x.shape for x in self.layerCache]
        print "L: ", [x.numberOfNeurons for x in self.layers]
        print "##########"