import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class NN:
    def __init__(self, network):
        self.network = np.asmatrix(network)

    def feed(self, _input):
        return sigmoid(self.network.T.dot(np.array(_input)))

    def eval(self, dataset):
        correct = 0
        # take all data and compare if predicted is correct
        for data in dataset:
            X = data[:-1]
            y = data[-1]

            # round for multiple classes
            res = self.feed(X)

            if res == y:
                correct += 1
        return float(correct) / len(dataset)
