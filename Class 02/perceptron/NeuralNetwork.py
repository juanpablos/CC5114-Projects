from perceptron.NeuronLayer import NeuronLayer


class NeuralNetwork:
    def __init__(self, neurons_first):
        self.firstLayer = None
        self.layers = [NeuronLayer(neurons_first)]
        self.lastLayer = None

    def add_layer(self, neuron_number):
        self.layers.append(NeuronLayer(neuron_number))

    def initialize(self, n_inputs):
        try:
            assert len(self.layers) > 1

            self.layers[0].initialize(n_inputs)
            for i in range(1, len(self.layers)):
                self.layers[i].initialize(self.layers[i-1].get_number_neurons())
            self.firstLayer = self.layers[0]
            self.lastLayer = self.layers[len(self.layers)-1]

        except AssertionError:
            print("Needs at least 2 layers.")

    def evaluate(self, inputs):
        print(self.firstLayer.evaluate(inputs))
