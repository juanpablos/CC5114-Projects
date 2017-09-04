import csv
import random
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from time import time
from Neurons.NeuralNetwork import NeuralNetwork
from Utils.utilities import generate_network, make_dataset, get_seeds
from Arbitrary_training import run_plot

matplotlib.style.use('ggplot')

available_data_types = ["layer", "neuron", "standard"]


# Generates a network with random values according to the current analysis that is being done
def get_network(data_type, variable, n_inputs, neurons_output, neurons_per_layer, layers):
    inputs = n_inputs
    outputs = neurons_output
    neurons = neurons_per_layer
    total_layers = layers

    if data_type == "layer":
        total_layers = variable
    elif data_type == "neuron":
        neurons = variable
    elif data_type == "standard":
        pass
    else:
        raise TypeError

    return generate_network(n_inputs=inputs, neurons_output=outputs,
                            neurons_per_layer=neurons, layers=total_layers)


# --------- DATA FUNCTIONS -----------


# Evaluates a network according to the input values.
# layers vs learning rate
# neurons vs learning rate
def get_analysis_data(train_set, test_set, train_expected, test_expected, output_file, data_type,
                      max_neurons_per_layer=3, total_layers=10,
                      epochs=1000, epoch_step=10, trials=100):
    # basic constraints
    assert total_layers > 0
    assert total_layers > 0
    assert isinstance(data_type, str)

    # zip the data lists
    dataset = make_dataset(train_set, train_expected)

    with open(output_file, 'w', newline="\n") as file:
        writer = csv.writer(file)
        # write the current analysis data type
        writer.writerow(["epoch", data_type, "correctness"])

        if data_type == "layer":
            max_variable = total_layers
        elif data_type == "neuron":
            max_variable = max_neurons_per_layer
        elif data_type == "standard":
            max_variable = 1
        else:
            raise TypeError

        for variable in range(1, max_variable + 1):

            a_network = get_network(data_type=data_type, variable=variable,
                                    n_inputs=len(train_set[0]), neurons_output=len(train_expected[0]),
                                    neurons_per_layer=max_neurons_per_layer, layers=total_layers)

            for epoch in range(1, epochs + epoch_step, epoch_step):
                network = NeuralNetwork(manual=True)
                network.initialize(network=a_network)

                print("{}: {} of {}\nepoch: {} of {}\n".format(data_type, variable, max_variable, epoch, epochs))
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
                writer.writerow([epoch, variable, correctness / trials])


# Real test with seeds dataset
# It is slow, better see the attached plots
def run_seeds_analysis(the_seeds_file, test_samples_class, iterations, output_file, shuffle=True):
    x_axis, network_correctness, total_error, _ = run_plot(the_seeds_file, test_samples_class, iterations, shuffle)
    with open(output_file, 'w', newline="\n") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "correctness", "error"])

        for x, y, e in zip(x_axis, network_correctness, total_error):
            writer.writerow([x, y, e])


# How fast data is analysed
def data_processing_time(data_set, data_expected, output_file, max_layers=5, max_neurons=5, epoch=1000):

    dataset = make_dataset(data_set, data_expected)

    with open(output_file, 'w', newline="\n") as file:
        writer = csv.writer(file)
        # write the current analysis data type
        writer.writerow(["layer", "neuron", "time"])

        for layer in range(1, max_layers + 1):
            for neuron in range(1, max_neurons + 1):

                a_network = generate_network(n_inputs=len(data_set[0]), neurons_output=len(data_set[0]),
                                             neurons_per_layer=neuron, layers=layer)

                network = NeuralNetwork(manual=True)
                network.initialize(network=a_network)

                print("layer: {} of {}\nneuron: {} of {}\n".format(layer, max_layers, neuron, max_neurons))

                ini = time()
                network.train_with_dataset(dataset=dataset, epoch=epoch)
                fin = time()

                writer.writerow([layer, neuron, (fin - ini)])


# Effects of different learning rates
# seeds dataset is used
def learning_rate_comparison(train_set, test_set, train_expected, test_expected, output_file, shuffle=True,
                             layers=2, neurons=5, max_learning_rate=2, learning_step=0.1, epochs=100, epoch_step=10):
    # zip the data lists
    dataset = make_dataset(train_set, train_expected)

    if shuffle:
        random.shuffle(dataset)

    with open(output_file, 'w', newline="\n") as file:
        writer = csv.writer(file)
        # write the current analysis data type
        writer.writerow(["epoch", "learning_rate", "correctness"])

        a_network = generate_network(n_inputs=len(train_set[0]), neurons_output=len(train_expected[0]),
                                     neurons_per_layer=neurons, layers=layers)

        for lr in np.arange(0.1, max_learning_rate + learning_step, learning_step):
            for epoch in range(1, epochs + epoch_step, epoch_step):
                network = NeuralNetwork(manual=True, learning_rate=lr)
                network.initialize(network=a_network)

                print("lr: {} of {}\nepoch: {} of {}\n".format(lr, max_learning_rate, epoch, epochs))
                # training
                network.train_with_dataset(dataset=dataset, epoch=epoch)

                # evaluation
                correctness = 0.
                for test_, exp in zip(test_set, test_expected):

                    ideal_output = exp
                    actual = network.feed(test_)
                    normalized_output = [round(e, 0) for e in actual]

                    if ideal_output == normalized_output:
                        correctness += 1.
                writer.writerow([epoch, lr, correctness / len(test_set)])


# --------- PLOT FUNCTIONS -----------


# Comparison with sorted vs shuffled data
def plot_data_shuffle_comparison(file_location1="Analysis/seeds_analysis_shuffled.csv",
                                 file_location2="Analysis/seeds_analysis_sorted.csv"):
    shuffled_seed = pd.read_csv(file_location1)
    sorted_seed = pd.read_csv(file_location2)

    shuffled_seed.columns = ['epoch', 'shuffled correctness', 'shuffled error']
    sorted_seed.columns = ['epoch', 'sorted correctness', 'sorted error']

    ax = shuffled_seed.plot(kind='line', x='epoch', y='shuffled correctness')
    shuffled_seed.plot(kind='line', x='epoch', y='shuffled error', secondary_y=True, ax=ax, style='-.')

    sorted_seed.plot(kind='line', x='epoch', y='sorted correctness', ax=ax)
    sorted_seed.plot(kind='line', x='epoch', y='sorted error', secondary_y=True, ax=ax, style='-.')

    ax.set_ylabel("Average precision")
    plt.ylabel("Error")

    plt.title("Comparison between seeds dataset - Shuffled and sorted")

    plt.show()


# Network layers impact on learning rate
def plot_hidden_impact(file_location="Analysis/hidden_impact.csv"):
    data_hidden = pd.read_csv(file_location)
    data_hidden.plot.scatter(x='epoch', y='layer', c='correctness', s=50, colormap='gist_rainbow',
                             title="Number of layers vs learning rate with 3 neurons per layer - XOR")
    plt.show()


# Neurons per layer impact on learning rate
def plot_neuron_impact(file_location="Analysis/neuron_impact.csv"):
    data_neuron = pd.read_csv(file_location)
    data_neuron.plot.scatter(x='epoch', y='neuron', c='correctness', s=50, colormap='gist_rainbow',
                             title="Number of neurons vs learning rate with 2 layers - XOR")
    plt.show()


# Plot of real case dataset - seeds dataset
def plot_seeds_data(file_location="Analysis/seeds_analysis.csv", shuffled=""):
    data_seed = pd.read_csv(file_location)
    data_seed.columns = ['epoch', 'correctness', 'error']

    ax = data_seed.plot(kind='line', x='epoch', y='correctness')
    data_seed.plot(kind='line', x='epoch', y='error', secondary_y=True, ax=ax, style='-.')

    ax.set_ylabel("Average precision")
    plt.ylabel("Error")

    s = ""
    if shuffled:
        s = "- " + shuffled

    plt.title("Precision and error for seeds dataset - 2 layers - 5 neurons {}".format(s))

    plt.show()


# Data processing plot
def plot_data_time(file_location="Analysis/layer_neuron_time.csv"):
    data_time = pd.read_csv(file_location)
    rs = data_time.pivot("layer", "neuron", "time")
    ax = sns.heatmap(rs, annot=True, cmap='jet', cbar_kws={'label': "Time [s]"})

    ax.invert_yaxis()

    plt.xlabel("Neurons")
    plt.ylabel("Layers")

    plt.title("Time taken for 1000 epoch training - XOR")

    plt.show()


# Effect of different learning rates
def plot_lr_epoch(file_location="Analysis/learning_rate_comp.csv"):
    data_lr = pd.read_csv(file_location)
    rs = data_lr.pivot("epoch", "learning_rate", "correctness")
    ax = sns.heatmap(rs, cmap='jet', cbar_kws={'label': "Average precision"})

    ax.invert_yaxis()

    plt.ylabel("Epoch")
    plt.xlabel("Learning Rate")

    plt.title("Precision for different learning rates - seeds dataset")

    plt.show()


# --------- DATASETS -----------

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

seed_train_set, seed_test_set, seed_train_expected, seed_test_expected = get_seeds("formatted_seeds.txt", 35)

# --------- FILES -----------

analysis_folder = "Analysis/"

hidden_file = analysis_folder + "hidden_impact.csv"
neuron_file = analysis_folder + "neuron_impact3.csv"

seeds_shuffled_file = analysis_folder + "seeds_analysis_shuffled.csv"
seeds_sorted_file = analysis_folder + "seeds_analysis_sorted.csv"

time_data = analysis_folder + "layer_neuron_time.csv"

lr_data_sorted = analysis_folder + "learning_rate_comp_sorted.csv"
lr_data_shuffled = analysis_folder + "learning_rate_comp_shuffled.csv"

# --------- GET DATA -----------

# get_analysis_data(train_set_XOR, test_set_XOR, train_expected_XOR, test_expected_XOR, output_file=hidden_file,
#                   data_type="layer", epochs=1000, total_layers=10, max_neurons_per_layer=3)
get_analysis_data(train_set_XOR, test_set_XOR, train_expected_XOR, test_expected_XOR, output_file=neuron_file,
                  data_type="neuron", epochs=1000, epoch_step=50, total_layers=2, max_neurons_per_layer=10)

# run_seeds_analysis("formatted_seeds.txt", 35, 100, seeds_shuffled_file)
# run_seeds_analysis("formatted_seeds.txt", 35, 100, seeds_sorted_file, shuffle=False)

# data_processing_time(train_set_XOR, train_expected_XOR, time_data, max_layers=10, max_neurons=10, epoch=1000)

# learning_rate_comparison(seed_train_set, seed_test_set, seed_train_expected, seed_test_expected, lr_data_sorted,
#                          shuffle=False, layers=2, neurons=5, max_learning_rate=2, learning_step=0.1, epochs=100)
# learning_rate_comparison(seed_train_set, seed_test_set, seed_train_expected, seed_test_expected, lr_data_shuffled,
#                          layers=2, neurons=5, max_learning_rate=2, learning_step=0.1, epochs=100)

# --------- PLOT -----------

# plot_hidden_impact(hidden_file)
plot_neuron_impact(neuron_file)
#
# plot_seeds_data(seeds_shuffled_file, "shuffled")
#
# plot_data_shuffle_comparison(seeds_shuffled_file, seeds_sorted_file)
#
# plot_data_time(time_data)
#
# plot_lr_epoch(lr_data_shuffled)
# plot_lr_epoch(lr_data_sorted)
