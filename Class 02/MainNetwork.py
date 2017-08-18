from Neurons.NeuralNetwork import NeuralNetwork
from Neurons.Neurons import Sigmoid
from Neurons.exceptions import UnmatchedLengthError

try:
    network = NeuralNetwork(neurons_first=30)
    network.add_layer(neuron_number=100)
    network.add_layer(neuron_number=10)
    network.add_layer(neuron_number=1)
    network.add_layer(neuron_number=5)

    network.rand_initialize(n_inputs=3)

    network.print_net_info()

    print("Network evaluation: {}".format(network.feed([1, 1, 1])))
except UnmatchedLengthError as e:
    print(e.error_args)
