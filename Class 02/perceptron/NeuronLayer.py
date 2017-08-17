import random

from perceptron.Neurons import Sigmoid


class NeuronLayer:
    def __init__(self, number):
        self.number = number
        self.neurons = list()

    def initialize(self, n_inputs):
        for _ in range(self.number):
            weight = []
            for i in range(n_inputs):
                weight.append(self.set_weight())
            bias = self.set_bias()
            self.neurons.append(Sigmoid(weight, bias))

    def set_weight(self):
        # TODO: randomize
        #return 4.
        return random.uniform(-4, 4)

    def set_bias(self):
        # TODO: randomize
        #return 1.
        return random.uniform(-4, 4)

    def get_number_neurons(self):
        return self.number

    def feed(self, inputs):
        output = list()
        print(inputs)
        for neuron in self.neurons:
            output.append(neuron.evaluate(inputs))
        return output

    def get_info(self):
        return "neurons: {}, weights: {}".format(len(self.neurons), len(self.neurons[0].weights))
