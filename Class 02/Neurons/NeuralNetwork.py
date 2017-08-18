from Neurons.NeuronLayer import NeuronLayer
from Neurons.exceptions import LayerError


class NeuralNetwork:
    def __init__(self, manual=False):
        self.firstLayer = None
        self.layers = list()
        self.lastLayer = None

        self.manual = manual
        self.initialized = False

    def add_layer(self, neuron_number):
        if not self.initialized:
            if self.manual:
                raise NotImplementedError("Sorry~~")
            else:
                self.layers.append(NeuronLayer(neuron_number))
        else:
            print("Network is already initialized. Cannot add another layer. Don't try.")

    def initialize(self, n_inputs=None):
        try:
            assert len(self.layers) > 1
            if self.manual:
                raise NotImplementedError("Sorry~~")
            else:
                self.layers[0].initialize(n_inputs)
                for i in range(1, len(self.layers)):
                    self.layers[i].initialize(self.layers[i - 1].get_number_neurons())
                self.firstLayer = self.layers[0]
                self.lastLayer = self.layers[len(self.layers) - 1]

            self.initialized = True

        except AssertionError:
            raise LayerError("Needs at least 2 layers.")

    def feed(self, inputs):
        out = self.firstLayer.feed(inputs)
        for layer in self.layers:
            if layer is not self.firstLayer:
                out = layer.feed(out)

        return out

    def print_net_info(self):
        for layer in self.layers:
            print("layer {}: {}".format(self.layers.index(layer), layer.get_info()))

    def train(self, inputs, expected_output):
        # STEP 1
        self.feed(inputs)

        # STEP 2
        # last
        self.lastLayer.update_last_deltas(expected_output=expected_output)

        # rest
        for i in range(len(self.layers) - 2, -1, -1):
            weights_collection = self.layers[i + 1].collect_weights(collect=self.layers[i].get_number_neurons())
            delta_collection = self.layers[i + 1].collect_deltas(collect=self.layers[i].get_number_neurons())
            self.layers[i].update_hidden_deltas(weights=weights_collection, deltas=delta_collection)

        # STEP 3
        # first
        self.firstLayer.update_neuron_weights(inputs=inputs)
        self.firstLayer.update_neuron_bias()

        for j in range(1, len(self.layers)):
            output_collection = self.layers[j - 1].collect_outputs()
            self.layers[j].update_neuron_weights(output_collection)
            self.layers[j].update_neuron_bias()

    def train_with_dataset(self, dataset, reps):
        for _ in range(reps):
            for data in dataset:
                self.train(data['inputs'], data['expected'])

    def print_deltas(self):
        print("-----deltas-----")
        for layer in self.layers:
            layer.print_deltas()
            print("\n")
        print("-----end deltas-----")

    def print_weights(self):
        print("-----weights-----")
        for layer in self.layers:
            layer.print_weights()
            print("\n")
        print("-----end weights-----")

        #
        #
        # def add_layer(self, neuron_number, weights=None):
        #     """
        #     Adds a new layer to the network.
        #     Needs the number of neurons present in the layer. Can include an optional parameter 'weights' which is a
        #     list of dictionaries representing the weights of each neuron.
        #     If both parameters are entered the length of 'weights' has to be 'neuron_number'.
        #
        #     :param neuron_number: the number of neurons in the layer.
        #     :param weights: optional weights of each neuron in the layer.
        #     """
        #     if not self.initialized:
        #         try:
        #             assert len(weights) == neuron_number
        #         except TypeError:
        #             pass
        #         except AssertionError:
        #             raise UnmatchedLengthError(weights=len(weights), inputs=neuron_number)
        #
        #
        #         # TODO
        #         new_layer = NeuronLayer(neuron_number, weights)
        #         self.layers.append(new_layer)
        #
        #         if not self.firstLayer:
        #             self.firstLayer = new_layer
        #     else:
        #         print("Network is already initialized. Cannot add another layer. Don't try.")
        #
        # def initialize(self, n_inputs_first=None):
        #     try:
        #         # more than one layer
        #         assert len(self.layers) > 1
        #         # TODO
        #         # first layer does not have weights
        #         if not self.firstLayer.has_set_weights():
        #             # the input has not been set!
        #             if not n_inputs_first:
        #                 raise LayerError("The first layer does not have any input data (number of inputs-weights).")
        #             # set the input with random weights
        #             else:
        #                 #TODO
        #                 self.firstLayer.rand_initialize(n_inputs_first)
        #
        #         for i in range(1, len(self.layers)):
        #             if not self.layers[i].has_set_weights():
        #             self.layers[i].initialize(self.layers[i - 1].get_number_neurons())
        #
        #                 layer.rand_init()
        #
        #     except AssertionError:
        #         raise LayerError("Needs at least 2 layers.")
