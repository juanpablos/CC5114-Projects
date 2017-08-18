from Neurons.NeuralNetwork import NeuralNetwork

network = NeuralNetwork()
network.add_layer(neuron_number=30)
network.add_layer(neuron_number=100)
network.add_layer(neuron_number=10)
network.add_layer(neuron_number=1)
network.add_layer(neuron_number=5)

network.initialize(n_inputs=4)

network.print_net_info()

out = network.feed([1, 1, 1, 10])
print("Network evaluation: {}".format(out))
