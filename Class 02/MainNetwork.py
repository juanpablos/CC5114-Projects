from perceptron.NeuralNetwork import NeuralNetwork

network = NeuralNetwork(neurons_first=2)
network.add_layer(neuron_number=1)

network.initialize(n_inputs=4)


network.evaluate([1,2,3,-1])
