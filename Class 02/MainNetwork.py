from Neurons.NeuralNetwork import NeuralNetwork

network = NeuralNetwork()
network.add_layer(neuron_number=3)
network.add_layer(neuron_number=1)

network.initialize(n_inputs=2)

# network.print_net_info()

network.print_weights()
network.print_bias()

a = network.train_with_dataset(dataset=[
    [[0., 0.], [0]],
    [[1., 1.], [0]],
    [[1., 0.], [1]],
    [[0., 1.], [1]]
], epoch=100)

print(a)

oneone = network.feed([1., 1.])[0]
onecero = network.feed([1., 0.])[0]
ceroone = network.feed([0., 1.])[0]
cerocero = network.feed([0., 0.])[0]

print("Training network")
print("1 AND 1 = {} actual output: {}".format(round(oneone, 0), oneone))
print("1 AND 0 = {} actual output: {}".format(round(onecero, 0), onecero))
print("0 AND 1 = {} actual output: {}".format(round(ceroone, 0), ceroone))
print("0 AND 0 = {} actual output: {}".format(round(cerocero, 0), cerocero))

done_network = \
    [
        [
            [2.57147557, 1.89695434, 0.380250607417],
            [6.78504447, -5.92964121, 2.97653335647],
            [4.41535064, -5.97567152, -1.96438569666]
        ],
        [
            [2.80838347, -9.39005226, 9.18965719, 2.00463814965]
        ]
    ]

network_trained = NeuralNetwork(manual=True)

network_trained.initialize(network=done_network)

network_trained.print_net_info()

oneone2 = network_trained.feed([1., 1.])[0]
onecero2 = network_trained.feed([1., 0.])[0]
ceroone2 = network_trained.feed([0., 1.])[0]
cerocero2 = network_trained.feed([0., 0.])[0]

print("Trained network")
print("1 AND 1 = {} actual output: {}".format(round(oneone2, 0), oneone2))
print("1 AND 0 = {} actual output: {}".format(round(onecero2, 0), onecero2))
print("0 AND 1 = {} actual output: {}".format(round(ceroone2, 0), ceroone2))
print("0 AND 0 = {} actual output: {}".format(round(cerocero2, 0), cerocero2))
