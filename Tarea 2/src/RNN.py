import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def derivative_sigmoid(z):
    return z * (1 - z)


def softmax(x, i):
    return np.exp(x[i]) / np.sum(np.exp(x[i]))


class RNN:
    def __init__(self, input_size, hidden_size, out_size, learning_rate):
        self.input_size = input_size
        self.hidden_layer_size = hidden_size
        self.output_size = out_size
        self.learning_rate = learning_rate

        self.w_xh = 2 * np.random.random((self.hidden_layer_size, self.input_size)) - 1
        self.w_hh = 2 * np.random.random((self.hidden_layer_size, self.hidden_layer_size)) - 1
        self.w_hy = 2* np.random.random((self.output_size, self.hidden_layer_size)) - 1

        self.b_h = np.zeros((self.hidden_layer_size, 1))
        self.b_y = np.zeros((self.output_size, 1))


    def forward_propagation(self, x):

        # time steps
        sequence = len(x)

        # states
        h = np.zeros((sequence + 1, self.hidden_layer_size))
        # outs
        y = np.zeros((sequence, self.output_size))

        #for the whole sequence of inputs
        for t in np.arange(sequence):
            # calculate state at time t. No problem with t=0 because python will take 0 in t-1, no index problem
            h[t] = np.tanh(np.dot(self.w_xh, x[t]) + np.dot(self.w_hh, h[t-1]) + self.b_h)
            y[t] = softmax(np.dot(self.w_hy, h[t]) + self.b_y, t)

        return [y, h]
