from Neurons.NeuralNetwork import NeuralNetwork

network = NeuralNetwork(neurons_first=30)
network.add_layer(neuron_number=100)
network.add_layer(neuron_number=10)
network.add_layer(neuron_number=1)
network.add_layer(neuron_number=5)

network.initialize(n_inputs=3)

network.print_net_info()

print("Network evaluation: {}".format(network.feed([1, 1, 1])))
