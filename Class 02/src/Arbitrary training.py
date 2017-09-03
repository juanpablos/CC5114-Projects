import csv
import random

import matplotlib
import matplotlib.pyplot as plt

from src.Neurons.NeuralNetwork import NeuralNetwork
from src.Utils.network_utilities import generate_network

matplotlib.style.use('ggplot')


def normalize(value, min_value, max_value, norm_l=0, norm_h=1):
    out = (value - min_value) * (norm_h - norm_l)
    out /= (max_value - min_value)
    out += norm_l
    return out


# Dont mind this...
def get_normalized_seeds(file):
    with open(file) as f:
        ff = csv.reader(f, delimiter=',')
        col1 = list()
        col2 = list()
        col3 = list()
        col4 = list()
        col5 = list()
        col6 = list()
        col7 = list()
        colexp = list()
        for row in ff:
            col1.append(float(row[0].strip('\t')))
            col2.append(float(row[1].strip('\t')))
            col3.append(float(row[2].strip('\t')))
            col4.append(float(row[3].strip('\t')))
            col5.append(float(row[4].strip('\t')))
            col6.append(float(row[5].strip('\t')))
            col7.append(float(row[6].strip('\t')))
            colexp.append(row[7].strip('\t'))
        minmax1 = [min(col1), max(col1)]
        minmax2 = [min(col2), max(col2)]
        minmax3 = [min(col3), max(col3)]
        minmax4 = [min(col4), max(col4)]
        minmax5 = [min(col5), max(col5)]
        minmax6 = [min(col6), max(col6)]
        minmax7 = [min(col7), max(col7)]

        _file = list()

        for i in range(len(col1)):
            row = list()
            row.append(normalize(col1[i], minmax1[0], minmax1[1]))
            row.append(normalize(col2[i], minmax2[0], minmax2[1]))
            row.append(normalize(col3[i], minmax3[0], minmax3[1]))
            row.append(normalize(col4[i], minmax4[0], minmax4[1]))
            row.append(normalize(col5[i], minmax5[0], minmax5[1]))
            row.append(normalize(col6[i], minmax6[0], minmax6[1]))
            row.append(normalize(col7[i], minmax7[0], minmax7[1]))

            if colexp[i] == '1':
                exp = [1, 0, 0]
            elif colexp[i] == '2':
                exp = [0, 1, 0]
            else:
                exp = [0, 0, 1]

            _file.append([row, exp])

        return _file


# Just works for the seeds dataset, but the idea is to divide the dataset
def split_seeds(formatted_array, number_test=40):
    train_set = list()
    train_expected = list()
    test_set = list()
    test_expected = list()

    counter = 0
    # assumes the classes are sorted
    for row in range(len(formatted_array)):

        if test_expected and test_expected[len(test_expected) - 1] != formatted_array[row][1]:
            counter = 0

        if counter >= number_test:
            arg, exp = formatted_array[row]
            train_set.append(arg)
            train_expected.append(exp)
        else:
            arg, exp = formatted_array[row]
            test_set.append(arg)
            test_expected.append(exp)

        counter += 1

    return train_set, test_set, train_expected, test_expected


def make_dataset(data, expected):
    dataset = list()
    for d in zip(data, expected):
        dataset.append(list(d))
    return dataset


def two_scales(ax1, x, data1, data2, c1, c2):
    ax2 = ax1.twinx()

    ax1.plot(x, data1, color=c1)
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel('Error')

    ax2.plot(x, data2, color=c2)
    ax2.set_ylabel('Average correct prediction')
    return ax1, ax2


def make_learning_graph(train_set, test_set, train_expected, test_expected, iterations=1000):
    dataset = list()
    for data in zip(train_set, train_expected):
        dataset.append(list(data))

    random.shuffle(dataset)

    network_correctness = list()
    total_error = list()

    a_network = generate_network(n_inputs=7, neurons_output=3, neurons_per_layer=5, layers=2)
    for iteration in range(1, iterations):
        network = NeuralNetwork(manual=True)
        network.initialize(network=a_network)
        # network.initialize(network=[
        #     [
        #         [0.95880457, 0.21784549, 0.02374389, 0.8078171, 0.77041552, 0.973901, 0.53887185, 0.8475814939725956],
        #         [0.90134008, 0.48191564, 0.55122769, 0.17294786, 0.41416922, 0.17605323, 0.34926146,
        #          0.5330750969278901],
        #       [0.05626469, 0.36670835, 0.30285952, 0.85846472, 0.34721228, 0.4840695, 0.87199855, 0.8328433612251267],
        #         [0.20923415, 0.85080592, 0.89696861, 0.62092947, 0.91565473, 0.81386364, 0.95018906,
        #          0.8208468096059854],
        #         [0.34147907, 0.70699459, 0.20382072, 0.67809213, 0.20533725, 0.865495, 0.42393388, 0.5547435115489855]
        #     ],
        #     [
        #         [0.46430968, 0.08246037, 0.70357265, 0.97389177, 0.9727195, 0.3342816691988084],
        #         [0.05044235, 0.48212749, 0.69009214, 0.92398632, 0.21715507, 0.9370159224603476],
        #         [0.77973832, 0.03933167, 0.68875107, 0.2456488, 0.29408093, 0.3281311894860519]
        #     ]
        # ])

        print(iteration)
        # training
        error = network.train_with_dataset(dataset=dataset, epoch=iteration)

        # evaluation
        correctness = 0
        for test_, exp in zip(test_set, test_expected):

            ideal_output = exp

            actual = network.feed(test_)
            normalized_output = [round(e, 0) for e in actual]

            if ideal_output == normalized_output:
                correctness += 1

        if error:
            total_error.append(error[-1])
        network_correctness.append(correctness / len(test_set))

    x_axis = list()
    for i in range(1, iterations):
        x_axis.append(i)

    fig, ax = plt.subplots()
    ax1, ax2 = two_scales(ax, x_axis, total_error, network_correctness, 'r', 'b')
    ax1.set_ylim([0, max(total_error)])

    plt.title("Seeds dataset with {} test iterations".format(len(test_set)))
    plt.show()


train, test, train_exp, test_exp = split_seeds(get_normalized_seeds("formatted_seeds.txt"), 35)

# This is reeeeeally slow, but shows a cool plot
make_learning_graph(train, test, train_exp, test_exp, iterations=50)
