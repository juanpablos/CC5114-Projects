
import matplotlib.pyplot as plt
import random

from Neurons.NeuralNetwork import NeuralNetwork

def make_learning_graph(iterations=1000, evaluation=100):
    network_correctness = list()

    for iteration in range(iterations):
        network = NeuralNetwork(manual=True)
        network.initialize(network=[
    [
        [0.52005262, 0.50090067, 0.6476664410034432],
        [0.6634167, 0.34459219, 0.934842553959054],
        [0.7176642, 0.41593824, 0.1908567125038647]
    ],
    [
        [0.05263594, 0.63529249, 0.89631397, 0.7987314051068816]
    ]
])

        print(iteration)
        # training
        network.train_with_dataset(dataset=[
            {'inputs': [1., 1.], 'expected': [0]},
            {'inputs': [0., 1.], 'expected': [1]},
            {'inputs': [1., 0.], 'expected': [1]},
            {'inputs': [0., 0.], 'expected': [0]}
        ], reps=iteration)


        # evaluation
        correctness = 0
        for test in range(evaluation):
            bool1 = random.randint(0, 1)
            bool2 = random.randint(0, 1)

            ideal_output = bool1^bool2

            actual = network.feed([bool1, bool2])
            normalized_output = round(actual[0], 0)

            if ideal_output == normalized_output:
                correctness += 1

        network_correctness.append(correctness / evaluation)

    x_axis = list()
    for i in range(iterations):
        x_axis.append(i)

    per_label, = plt.plot(x_axis, network_correctness, 'b', alpha=0.5, label='Network training')
    plt.legend(handles=[per_label])
    axes = plt.gca()
    axes.set_ylim([0, 1])
    plt.xlabel('Number of iterations')
    plt.ylabel('Average correct prediction')
    plt.show()


make_learning_graph(iterations=1000)