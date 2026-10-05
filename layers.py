from perceptron import perceptron
from activation_function import phi

inputs = [0.1, 0.1, 0.2]
weights = [0.2, 0.5, 0.5, 0.5]
neuron = perceptron(inputs, weights, phi.sigmoid)
print(neuron.output)