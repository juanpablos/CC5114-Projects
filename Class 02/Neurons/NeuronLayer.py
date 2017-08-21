import random

import numpy as np

from Neurons.Neurons import Sigmoid


def set_rand_weights():
    return random.random()


def set_rand_bias():
    return random.random()


class NeuronLayer:
    def __init__(self, number=None):

        self.number = number
        self.neurons = list()

    def initialize(self, n_inputs=None, neuron_weights=None):
        if neuron_weights:
            self.number = len(neuron_weights)
            # TODO: check constraints
            for w in neuron_weights:
                self.neurons.append(Sigmoid(w[:-1], w[-1]))
            return self
        else:
            # TODO: check for inputs not None
            for _ in range(self.number):
                weights = []
                for i in range(n_inputs):
                    weights.append(set_rand_weights())
                bias = set_rand_bias()
                self.neurons.append(Sigmoid(weights, bias))

    def get_number_neurons(self):
        return self.number

    def feed(self, inputs):
        output = list()
        for neuron in self.neurons:
            output.append(neuron.evaluate(inputs))
        return output

    def get_info(self):
        return "neurons: {}, weights: {}".format(len(self.neurons), len(self.neurons[0].weights))

    def update_last_deltas(self, expected_output):
        for i in range(len(expected_output)):
            error = expected_output[i] - self.neurons[i].output
            self.neurons[i].update_delta(error=error)

    def collect_weights(self, collect):
        weights = list()
        for i in range(collect):
            i_weights = list()
            for neuron in self.neurons:
                i_weights.append(neuron.weights[i])
            weights.append(i_weights)
        return weights

    def collect_deltas(self, collect):
        deltas = list()
        for i in range(collect):
            i_deltas = list()
            for neuron in self.neurons:
                i_deltas.append(neuron.delta)
            deltas.append(i_deltas)
        return deltas

    def collect_outputs(self):
        outputs = list()
        for neuron in self.neurons:
            outputs.append(neuron.output)
        return outputs

    def update_hidden_deltas(self, weights, deltas):
        for i in range(len(self.neurons)):
            error = sum(np.array(weights[i]) * np.array(deltas[i]))
            self.neurons[i].update_delta(error=error)

    def update_neuron_weights(self, inputs):
        for neuron in self.neurons:
            neuron.update_weights(inputs)

    def update_neuron_bias(self):
        for neuron in self.neurons:
            neuron.update_bias()

    def print_deltas(self):
        for neuron in self.neurons:
            print(neuron.delta)

    def print_weights(self):
        for neuron in self.neurons:
            print(neuron.weights)

    def print_bias(self):
        for neuron in self.neurons:
            print(neuron.bias)