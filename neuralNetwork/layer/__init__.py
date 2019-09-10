



class Layer(object):
    def __init__(self, numberOfNeurons, activation):
        """Neural Network Layer
        
        Arguments:
            numberOfNeurons {int} -- Number of neurons in this layer
            activation {Activation} -- Activation function of the layer
        """
        self.numberOfNeurons = numberOfNeurons
        self.activation = activation
