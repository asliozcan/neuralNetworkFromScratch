"""



"""
import numpy as np


class FeedForwardNN(object):
    def __init__(self, inputSize, layers=[], costFunction=None):
        """Feed Forward Neural Network Object     
        
        Arguments:
            object {[type]} -- 
            inputSize {int} -- 
            layers {Layer[]} -- Array of layers which are included in neural network
            costFunction {CostFunction} -- Cost function of neural network
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
        """[summary]
        
        Arguments:
            trainX {[type]} -- [description]
            trainY {[type]} -- [description]
        
        Keyword Arguments:
            learningRate {float} -- [description] (default: {0.001})
            numberOfEpoch {int} -- [description] (default: {10})
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
        """[summary]
        
        Arguments:
            input {[type]} -- [description]
        """
        r = self._forward(x)
        return r 


    def save(self, filename):
        """[summary]
        
        Arguments:
            filename {[type]} -- [description]
        """
    
    def laod(self, filename):
        """[summary]
        
        Arguments:
            filename {[type]} -- [description]
        """

    def _forward(self, x):
        self.layerCache=[]
        for i in range(0, len(self.layers)):
            #raw_input("wait forw")
            self.layerCache.append(x)
            x = self.layers[i].activation.calc( np.dot( x , self._weights[i] ) )
        return x

    def _backprop(self, y, yPredict):
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
        print "##########"
        print "W: ", [x.shape for x in self._weights]
        print "derivatives: ", [x.shape for x in self.d_weights]
        print "LC: ", [x.shape for x in self.layerCache]
        print "L: ", [x.numberOfNeurons for x in self.layers]
        print "##########"