import random

from Neurons.Neurons import Sigmoid


class NeuronLayer:
    def __init__(self, number):
        self.number = number
        self.neurons = list()

    def initialize(self, n_inputs):
        for _ in range(self.number):
            weights = []
            for i in range(n_inputs):
                weights.append(self.set_rand_weights())
            bias = self.set_rand_bias()
            self.neurons.append(Sigmoid(weights, bias))

    def set_rand_weights(self):
        return random.uniform(-4, 4)

    def set_rand_bias(self):
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
