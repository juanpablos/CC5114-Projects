from perceptron.Neurons import Sigmoid


class NeuronLayer:
    def __init__(self, number):
        self.number = number
        self.neurons = list()

    def initialize(self, n_inputs):
        for _ in range(self.number):
            weight = []
            for i in range(n_inputs):
                weight.append(self.setWeight())
            bias = self.setBias()
            self.neurons.append(Sigmoid(weight, bias))

    def setWeight(self):
        return 4

    def setBias(self):
        return 1

    def get_number_neurons(self):
        return self.number

    def evaluate(self, inputs):
        output = list()
        for neuron in self.neurons:
            output.append(neuron.evaluate(inputs))
        return output
