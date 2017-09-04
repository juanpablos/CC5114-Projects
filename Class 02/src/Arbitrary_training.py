import random

import matplotlib
import matplotlib.pyplot as plt

from Neurons.NeuralNetwork import NeuralNetwork
from Utils.utilities import generate_network, get_seeds

matplotlib.style.use('ggplot')


def two_scales(ax1, x, data1, data2, c1, c2):
    ax2 = ax1.twinx()

    ax1.plot(x, data1, color=c1)
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel('Error')

    ax2.plot(x, data2, color=c2)
    ax2.set_ylabel('Average correct prediction')
    return ax1, ax2


def make_learning_graph(train_set, test_set, train_expected, test_expected, iterations=1000, shuffle=True):
    dataset = list()
    for data in zip(train_set, train_expected):
        dataset.append(list(data))

    if shuffle:
        random.shuffle(dataset)

    network_correctness = list()
    total_error = list()

    a_network = generate_network(n_inputs=7, neurons_output=3, neurons_per_layer=5, layers=2)
    for iteration in range(1, iterations):
        network = NeuralNetwork(manual=True)
        network.initialize(network=a_network)

        print(iteration)
        # training
        error = network.train_with_dataset(dataset=dataset, epoch=iteration)

        # evaluation
        correctness = 0.
        for test_, exp in zip(test_set, test_expected):

            ideal_output = exp

            actual = network.feed(test_)
            normalized_output = [round(e, 0) for e in actual]

            if ideal_output == normalized_output:
                correctness += 1.

        if error:
            total_error.append(error[-1])
        network_correctness.append(correctness / len(test_set))

    x_axis = list()
    for i in range(1, iterations):
        x_axis.append(i)

    return x_axis, network_correctness, total_error


def run_plot(seeds_file, test_sample_class, iterations, shuffle=True):
    train, test, train_exp, test_exp = get_seeds(seeds_file, test_sample_class)

    # This is reeeeeally slow, but shows a cool plot
    x_axis, network_correctness, total_error = make_learning_graph(train, test, train_exp, test_exp,
                                                                   iterations=iterations, shuffle=shuffle)

    return x_axis, network_correctness, total_error, len(test)


def main():
    x_axis, network_correctness, total_error, test = run_plot("formatted_seeds.txt", 35, 100)
    fig, ax = plt.subplots()

    ax1, ax2 = two_scales(ax, x_axis, total_error, network_correctness, 'r', 'b')
    ax1.set_ylim([0, max(total_error)])
    plt.title("Seeds dataset with {} test iterations".format(test))
    plt.show()


if __name__ == '__main__':
    main()
