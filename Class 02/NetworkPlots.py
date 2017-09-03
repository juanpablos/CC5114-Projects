import random

import matplotlib.pyplot as plt

from Neurons.NeuralNetwork import NeuralNetwork
from network_utilities import generate_network


def make_learning_graph(train_set, test_set, train_expected, test_expected, iterations=1000, evaluation=100):
    dataset = list()
    for data in zip(train_set, train_expected):
        dataset.append(list(data))

    network_correctness = list()
    total_error = list()

    a_network = generate_network(n_inputs=len(train_set[0]), neurons_output=len(train_expected[0]),
                                 neurons_per_layer=3, layers=3)
    for iteration in range(1, iterations, 10):
        network = NeuralNetwork(manual=True)
        network.initialize(network=a_network)
        # network = NeuralNetwork(manual=True)
        # network.initialize(network=[
        #     [
        #         [0.52005262, 0.50090067, 0.6476664410034432],
        #         [0.6634167, 0.34459219, 0.934842553959054],
        #         [0.7176642, 0.41593824, 0.1908567125038647]
        #     ],
        #     [
        #         [0.52005262, 0.50090067, 0.50090067, 0.6476664410034432],
        #         [0.6634167, 0.34459219, 0.50090067, 0.934842553959054],
        #         [0.7176642, 0.41593824, 0.50090067, 0.1908567125038647]
        #     ],
        #     [
        #         [0.05263594, 0.63529249, 0.89631397, 0.7987314051068816],
        #         [0.52005262, 0.63529249, 0.50090067, 0.6476664410034432]
        #     ]
        # ])

        print(iteration)
        # training
        error = network.train_with_dataset(dataset=dataset, epoch=iteration)

        # evaluation
        correctness = 0.
        tests = list(zip(test_set, test_expected))
        for test in range(evaluation):
            _test = random.choice(tests)
            ideal_output = _test[1]
            actual = network.feed(_test[0])
            normalized_output = [round(e, 0) for e in actual]
            if ideal_output == normalized_output:
                correctness += 1.
        if error:
            total_error.append(error[-1])
        network_correctness.append(correctness / evaluation)

    x_axis = list()
    for i in range(1, iterations, 10):
        x_axis.append(i)

    #total_error.insert(1, total_error[0])

    per_label, = plt.plot(x_axis, network_correctness, 'b', alpha=0.5, label='Network training')
    error_label, = plt.plot(x_axis, total_error, 'r', alpha=0.5, label='Network error')
    plt.legend(handles=[per_label, error_label])
    axes = plt.gca()
    axes.set_ylim([0, max(total_error) + 1])
    plt.xlabel('Epochs')
    plt.ylabel('Average correct prediction')
    plt.title('Training of XOR')
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

make_learning_graph(train_set_XOR, test_set_XOR, train_expected_XOR, test_expected_XOR, iterations=600)
