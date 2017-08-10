import random
from perceptron.Neurons import Perceptron, Sigmoid
import matplotlib.pyplot as plt


def f(an_x):
    return an_x*4 + 1


def make_visual_graph(a_neuron, canvas, training_iterations, showing_points, threshold=1):

    x_function = list()
    y_function = list()

    x_red_point = list()
    y_red_point = list()
    x_blue_point = list()
    y_blue_point = list()

    for _ in range(training_iterations):
        rand_x = random.uniform(-canvas, canvas)
        rand_y = random.uniform(-canvas, canvas)

        ideal_output = 0
        if f(rand_x) < rand_y:
            ideal_output = 1

        a_neuron.train([rand_x, rand_y], ideal_output)

    for i in range(0, showing_points):
        x = random.uniform(-canvas, canvas)
        y = random.uniform(-canvas, canvas)

        output = a_neuron.evaluate([x, y])

        if output >= threshold:
            x_red_point.append(x)
            y_red_point.append(y)
        else:
            x_blue_point.append(x)
            y_blue_point.append(y)

    for j in range(2*int(-canvas/3), 2*int(canvas/3)):
        x_function.append(j)
        y_function.append(f(j))

    plt.plot(x_red_point, y_red_point, 'ro', x_blue_point, y_blue_point, 'bo', x_function, y_function, 'g')
    plt.show()


def make_learning_graph(canvas, iterations=200, calc_points=100, threshold=1.):

    perceptron_training_correctness = list()
    sigmoid_training_correctness = list()

    for iteration in range(iterations):
        perceptron = Perceptron([1., 5.], -1)
        sigmoid = Sigmoid([1., 5.], -1, 0.5)

        # training
        for train in range(iteration):
            rand_x = random.uniform(-canvas, canvas)
            rand_y = random.uniform(-canvas, canvas)

            ideal_output = 0
            if f(rand_x) < rand_y:
                ideal_output = 1

            perceptron.train([rand_x, rand_y], ideal_output)
            sigmoid.train([rand_x, rand_y], ideal_output)

        #evaluation
        correct_points_perceptron = 0
        correct_points_sigmoid = 0
        for point in range(calc_points):
            rand_x = random.uniform(-canvas, canvas)
            rand_y = random.uniform(-canvas, canvas)

            ideal_output = 0
            if f(rand_x) < rand_y:
                ideal_output = 1

            perceptron_output = perceptron.evaluate([rand_x, rand_y])
            sigmoid_output = sigmoid.evaluate([rand_x, rand_y])
            normalized_output = 1
            if sigmoid_output < threshold:
                normalized_output = 0

            if ideal_output == normalized_output:
                correct_points_sigmoid += 1
            if ideal_output == perceptron_output:
                correct_points_perceptron += 1

        perceptron_training_correctness.append(correct_points_perceptron / calc_points)
        sigmoid_training_correctness.append(correct_points_sigmoid / calc_points)

    x_axis = list()
    for i in range(iterations):
        x_axis.append(i)

    plt.plot(x_axis, perceptron_training_correctness, 'b', x_axis, sigmoid_training_correctness, 'r', alpha=0.5)
    axes = plt.gca()
    axes.set_ylim([0, 1])
    plt.show()


# good curve
make_learning_graph(50, iterations=1000, threshold=1.)
# no
make_learning_graph(50, iterations=200, threshold=0.5)
# no
make_learning_graph(50, iterations=200, threshold=0.2)
