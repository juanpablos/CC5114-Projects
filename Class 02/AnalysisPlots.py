import csv
import random
import pandas as pd
from Neurons.NeuralNetwork import NeuralNetwork
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')


def get_rand():
    return random.random()


def prettify_network(a_network):
    for layer in a_network:
        for neuron in layer:
            print(neuron)
        print("")


# generates a random network with the input parameters
def generate_network(n_inputs, neurons_output, neurons_per_layer, layers):
    a_network = list()

    first_layer = list()
    # input
    # input is number of weights in initial layer
    for input_n in range(neurons_per_layer):
        input_neuron = list()
        for n_input in range(n_inputs + 1):
            input_neuron.append(get_rand())
        first_layer.append(input_neuron)
    a_network.append(first_layer)

    # rest
    for layer in range(layers - 2):
        a_layer = list()
        for neuron in range(neurons_per_layer):
            a_neuron = list()
            for _ in range(neurons_per_layer + 1):
                a_neuron.append(get_rand())
            a_layer.append(a_neuron)
        a_network.append(a_layer)

    last_layer = list()
    # out
    # out is number of neurons in outer later
    for out_n in range(neurons_output):
        output_neuron = list()
        for n_out in range(neurons_per_layer + 1):
            output_neuron.append(get_rand())
        last_layer.append(output_neuron)
    a_network.append(last_layer)

    return a_network


# Real test with seeds dataset

# Impact of hidden layers in learning rate
def impact_hidden_layer(train_set, test_set, train_expected, test_expected, total_layers=10, epochs=1000,
                        epoch_step=10, trials=100, output_file="hidden_impact.csv"):

    assert total_layers > 1

    dataset = list()
    for data in zip(train_set, train_expected):
        dataset.append(list(data))

    with open(output_file, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "layer", "correctness"])

        for layer in range(2, total_layers + 1):
            network = NeuralNetwork(manual=True)
            network.initialize(network=generate_network(n_inputs=2, neurons_output=2,
                                                        neurons_per_layer=3, layers=layer))

            for epoch in range(1, epochs, epoch_step):
                print("layer: {}\nepoch: {}\n".format(layer, epoch))
                # training
                network.train_with_dataset(dataset=dataset, epoch=epoch)

                # evaluation
                correctness = 0.
                tests = list(zip(test_set, test_expected))
                for t in range(trials):
                    _test = random.choice(tests)

                    ideal_output = _test[1]
                    actual = network.feed(_test[0])
                    normalized_output = [round(e, 0) for e in actual]
                    if ideal_output == normalized_output:
                        correctness += 1.
                writer.writerow([epoch, layer, correctness / trials])


# TODO
# Impact of number of neurons in learning rate

# TODO
# How fast data is analysed

# TODO
# Effects of different learning rates

# TODO
# Comparison with sorted vs shuffled data


def plot_hidden_impact(file_location="Analysis/hidden_impact.csv"):
    data_hidden = pd.read_csv(file_location)
    data_hidden.plot.scatter(x='epoch', y='correctness', c='layer', s=50, colormap='gist_rainbow',
                             title="Number of hidden layers vs learning rate")
    plt.show()


train_set_XOR = [
    [0., 0.],
    [1., 1.],
    [1., 0.],
    [0., 1.]
]
train_expected_XOR = [
    [1, 0],
    [1, 0],
    [0, 1],
    [0, 1]
]
test_set_XOR = [
    [0., 0.],
    [1., 1.],
    [1., 0.],
    [0., 1.]
]
test_expected_XOR = [
    [1, 0],
    [1, 0],
    [0, 1],
    [0, 1]
]

hidden_file = "Analysis/hidden_impact.csv"
# impact_hidden_layer(train_set_XOR, test_set_XOR, train_expected_XOR,
# test_expected_XOR, epochs=1000, total_layers=10, output_file=hidden_file)

plot_hidden_impact(hidden_file)
