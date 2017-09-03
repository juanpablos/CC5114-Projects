import csv
import random

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from src.Neurons.NeuralNetwork import NeuralNetwork
from src.Utils.network_utilities import generate_network

matplotlib.style.use('ggplot')


# Real test with seeds dataset

# Impact of hidden layers in learning rate
def impact_hidden_layer(train_set, test_set, train_expected, test_expected, neuron_layer=3, total_layers=10,
                        epochs=1000,
                        epoch_step=10, trials=100, output_file="hidden_impact.csv"):
    assert total_layers > 0

    dataset = list()
    for data in zip(train_set, train_expected):
        dataset.append(list(data))

    with open(output_file, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "layer", "correctness"])

        for layer in range(1, total_layers + 1):

            a_network = generate_network(n_inputs=len(train_set[0]), neurons_output=len(train_expected[0]),
                                         neurons_per_layer=neuron_layer, layers=layer)

            for epoch in range(1, epochs, epoch_step):
                network = NeuralNetwork(manual=True)
                network.initialize(network=a_network)

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


# Impact of number of neurons in learning rate
def impact_number_neuron(train_set, test_set, train_expected, test_expected, max_neurons_per_layer=100, layers=2,
                         epochs=1000, epoch_step=10, trials=100, output_file="neuron_impact.csv"):
    assert max_neurons_per_layer > 0

    dataset = list()
    for data in zip(train_set, train_expected):
        dataset.append(list(data))

    with open(output_file, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "neuron_number", "correctness"])

        for neuron in range(1, max_neurons_per_layer + 1):

            a_network = generate_network(n_inputs=len(train_set[0]), neurons_output=len(train_expected[0]),
                                         neurons_per_layer=neuron, layers=layers)

            for epoch in range(1, epochs, epoch_step):
                network = NeuralNetwork(manual=True)
                network.initialize(network=a_network)

                print("neuron: {}\nepoch: {}\n".format(neuron, epoch))
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
                writer.writerow([epoch, neuron, correctness / trials])


# TODO
# How fast data is analysed

# TODO
# Effects of different learning rates7

# TODO
# Comparison with sorted vs shuffled data


def plot_hidden_impact(file_location="Analysis/hidden_impact.csv"):
    data_hidden = pd.read_csv(file_location)
    data_hidden.plot.scatter(x='epoch', y='layer', c='correctness', s=50, colormap='gist_rainbow',
                             title="Number of layers vs learning rate with 3 neurons per layer")
    plt.show()


def plot_neuron_impact(file_location="Analysis/neuron_impact.csv"):
    data_neuron = pd.read_csv(file_location)
    data_neuron.plot.scatter(x='epoch', y='neuron_number', c='correctness', s=50, colormap='gist_rainbow',
                             title="Number of neurons vs learning rate with 2 layers")
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
# neuron_file = "Analysis/neuron_impact.csv"
#
impact_hidden_layer(train_set_XOR, test_set_XOR, train_expected_XOR, test_expected_XOR,
                    epochs=1000, total_layers=10, neuron_layer=3, output_file=hidden_file)
#
# impact_number_neuron(train_set_XOR, test_set_XOR, train_expected_XOR, test_expected_XOR,
#                      epochs=1000, max_neurons_per_layer=20, output_file=neuron_file)

# data_hidden = pd.read_csv(neuron_file)
# data_hidden.plot.scatter(x='epoch', y='correctness', c='neuron_number', s=50, colormap='gist_rainbow')
# plt.show()

plot_hidden_impact(hidden_file)

# plot_neuron_impact(neuron_file)
