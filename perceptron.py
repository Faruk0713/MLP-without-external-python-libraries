from activation_function import phi
class perceptron:

    def __init__(self, inputs, parameters, activation):
        self.inputs = inputs
        self.activation = activation        
        self.weights = parameters[1:]
        self.bias = parameters[0]
        vk = self.bias
        for i in range(len(self.inputs)):
            vk = vk + self.inputs[i]*self.weights[i]
        self.output = self.activation(vk)
        
    def show_params(self):
        print("Bias = ", self.bias)
        print("weights = ")
        for i in range(len(self.weights)):
            print(self.weights[i],)






    