from Neurons.NeuronLayer import NeuronLayer
from Neurons.exceptions import UnmatchedLengthError


class NeuralNetwork:

    def __init__(self):
        self.firstLayer = None
        self.layers = list()
        self.lastLayer = None

        self.initialized = False

    def add_layer(self, neuron_number, weights=None):
        """
        Adds a new layer to the network.
        Needs the number of neurons present in the layer. Can include an optional parameter 'weights' which is a
        list of lists representing the weights of each neuron.
        If both parameters are entered the length of 'weights' has to be 'neuron_number'.

        :param neuron_number: the number of neurons in the layer.
        :param weights: optional weights of each neuron in the layer.
        """
        try:
            assert len(weights) == neuron_number
        except TypeError:
            pass
        except AssertionError:
            raise UnmatchedLengthError(weights=len(weights), inputs=neuron_number)

        self.layers.append(NeuronLayer(neuron_number, weights))









    def rand_initialize(self, n_inputs):
        try:
            assert len(self.layers) > 1

            self.layers[0].initialize(n_inputs)
            for i in range(1, len(self.layers)):
                self.layers[i].initialize(self.layers[i - 1].get_number_neurons())
            self.firstLayer = self.layers[0]
            self.lastLayer = self.layers[len(self.layers) - 1]

        except AssertionError:
            print("Needs at least 2 layers.")

    def feed(self, inputs):
        out = self.firstLayer.feed(inputs)
        for layer in self.layers:
            if layer is not self.firstLayer:
                out = layer.feed(out)

        return out

    def print_net_info(self):
        for layer in self.layers:
            print("layer {}: {}".format(self.layers.index(layer), layer.get_info()))
