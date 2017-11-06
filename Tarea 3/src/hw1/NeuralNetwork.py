from src.hw1.NeuronLayer import NeuronLayer
from src.hw1.exceptions import LayerError


class NeuralNetwork:
    def __init__(self, manual=False, learning_rate=0.5):
        self.firstLayer = None
        self.layers = list()
        self.lastLayer = None

        self.manual = manual
        self.initialized = False

        self.learning_rate = learning_rate

    def add_layer(self, neuron_number):
        if not self.initialized:
            if self.manual:
                print("You are on manual, enter a full network to/in initialize.")
            else:
                self.layers.append(NeuronLayer(number=neuron_number, learning_rate=self.learning_rate))
        else:
            print("Network is already initialized. Cannot add another layer. Don't try.")

    def initialize(self, n_inputs=None, network=None):
        if not self.manual:
            try:
                assert len(self.layers) > 0

                self.layers[0].initialize(n_inputs)
                for i in range(1, len(self.layers)):
                    self.layers[i].initialize(self.layers[i - 1].get_number_neurons())
                self.firstLayer = self.layers[0]
                self.lastLayer = self.layers[len(self.layers) - 1]

                self.initialized = True

            except AssertionError:
                raise LayerError("Needs at least 1 layers.")
        else:
            try:
                assert network is not None
                assert len(network) > 0
                if len(network) > 1:
                    # check weights in last layer corresponds to neurons in the layer before
                    assert (len(network[-1][0]) - 1) == (len(network[-2]))

                for layer in network:
                    self.layers.append(NeuronLayer(learning_rate=self.learning_rate).initialize(neuron_weights=layer))
                self.firstLayer = self.layers[0]
                self.lastLayer = self.layers[len(self.layers) - 1]

                self.initialized = True
            except AssertionError:
                raise LayerError("The passed network is not valid.")

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
        output = self.feed(inputs)

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

        return sum([(expected_output[i] - output[i]) ** 2 for i in range(len(expected_output))])

    def train_with_dataset(self, dataset, epoch):
        error = list()
        for _ in range(epoch):
            local_error = 0
            for data in dataset:
                local_error += self.train(data[0], data[1])
            error.append(local_error)
        return error

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

    def print_bias(self):
        print("-----bias-----")
        for layer in self.layers:
            layer.print_bias()
            print("\n")
        print("-----end bias-----")
