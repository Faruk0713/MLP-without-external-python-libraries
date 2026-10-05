from perceptron import perceptron
from activation_function import phi

class InputLayer:

    def __init__(self, n):
        self.layer = []
        for i in range(n):
            neuron = perceptron([1], [1], phi.linear)
            self.layer.append(neuron)
        self.output = []
        for i in range(n):
            self.output.append(self.layer[i].output)
        
class HiddenLayer:

    def __init__(self, n, previous_layer):
        self.layer = []
        for i in range(n):
            neuron = perceptron(previous_layer.output, [0.5]*(len(previous_layer.output)+1), phi.sigmoid)
            self.layer.append(neuron)
        self.output = []
        for i in range(n):
            self.output.append(self.layer[i].output)


class OutputLayer:
    def __init__(self, previous_layer):
        neuron = perceptron(previous_layer.output, [0.5]*(len(previous_layer.output)+1), phi.sigmoid)
        self.output = neuron.output