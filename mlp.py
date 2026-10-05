from activation_function import phi
from perceptron import perceptron
from layers import InputLayer, HiddenLayer, OutputLayer

class MultiLayerPerceptron:
    
    def __init__(self, inputs, hidden_layers, perceptrons = []):

        self.Layer1 = InputLayer(inputs)
        
        self.hiddenlayer = []
        
        self.Layer2 = HiddenLayer(perceptrons[0], self.Layer1)        
        self.hiddenlayer.append(self.Layer2)

        for i in range(hidden_layers-1):
            layer = HiddenLayer(perceptrons[i+1], self.hiddenlayer[i])
            self.hiddenlayer.append(layer)

        self.outputlayer = OutputLayer(self.hiddenlayer[hidden_layers-1])
        self.mlp_output = self.outputlayer.output