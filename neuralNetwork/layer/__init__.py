



class Layer(object):
    def __init__(self, numberOfNeurons, activation):
        """Neural Network Layer
        
            :param numberOfNeurons: int: -- Number of neurons in this layer
            :param activation: Activation: -- Activation function of the layer
        """
        self.numberOfNeurons = numberOfNeurons
        self.activation = activation
