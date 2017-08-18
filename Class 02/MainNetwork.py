from Neurons.NeuralNetwork import NeuralNetwork

network = NeuralNetwork()
network.add_layer(neuron_number=2)
network.add_layer(neuron_number=1)

network.initialize(n_inputs=2)

network.print_net_info()

out = network.feed([1., 0.])
print("Network evaluation: {}".format(out))

#network.print_deltas()
#network.print_weights()
for _ in range(10):
    network.train(inputs=[1., 1.], expected_output=[1])
    network.train(inputs=[0., 1.], expected_output=[0])
    network.train(inputs=[1., 1.], expected_output=[1])
    network.train(inputs=[0., 1.], expected_output=[0])
    network.train(inputs=[1., 1.], expected_output=[1])
    network.train(inputs=[0., 0.], expected_output=[0])
    network.train(inputs=[1., 1.], expected_output=[1])
    network.train(inputs=[1., 0.], expected_output=[0])
    network.train(inputs=[1., 1.], expected_output=[1])
    network.train(inputs=[0., 0.], expected_output=[0])


network.print_deltas()
network.print_weights()

oneone = network.feed([1., 1.])[0]
onecero = network.feed([1., 0.])[0]
ceroone = network.feed([0., 1.])[0]
cerocero = network.feed([0., 0.])[0]

print("1 AND 1 = {} with prob being one: {}".format(round(oneone, 0), oneone))
print("1 AND 0 = {} with prob being one: {}".format(round(onecero, 0), onecero))
print("0 AND 1 = {} with prob being one: {}".format(round(ceroone, 0), ceroone))
print("0 AND 0 = {} with prob being one: {}".format(round(cerocero, 0), cerocero))
