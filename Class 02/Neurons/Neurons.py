import numpy as np

from .exceptions import UnmatchedLengthError

learning_rate = 0.01


class Perceptron:
    def __init__(self, weight_list, bias):
        self.weights = np.array(weight_list)
        self.bias = bias

    def evaluate(self, input_list):
        try:
            assert len(self.weights) == len(input_list)

            res_sum = sum(self.weights * np.array(input_list))
            res = 0
            if res_sum + self.bias > 0:
                res = 1
            return res

        except AssertionError:
            print("The length of the input defers from the weights vector.")
        except TypeError:
            print("You either entered a non number list in constructor or input. Types don't match.")

    def train(self, input_train_list, expected_result):
        local_result = self.evaluate(input_train_list)
        if local_result > expected_result:
            self.decrease_weights(input_train_list)
        elif local_result < expected_result:
            self.increase_weights(input_train_list)

    def decrease_weights(self, inputs):
        self.weights -= np.array(inputs) * learning_rate

    def increase_weights(self, inputs):
        self.weights += np.array(inputs) * learning_rate


class Sigmoid(Perceptron):
    def __init__(self, weight_list, bias, threshold=0.5):
        super().__init__(weight_list, bias)
        self.threshold = threshold

    def evaluate(self, input_list):
        try:
            assert len(self.weights) == len(input_list)

            res_sum = sum(self.weights * np.array(input_list)) + self.bias
            return self.activation_function(res_sum)

        except AssertionError:
            raise UnmatchedLengthError(weights=len(self.weights), inputs=len(input_list))
        except TypeError:
            print("You either entered a non number list in constructor or input. Types don't match.")

    def train(self, input_train_list, expected_result):
        local_result = self.evaluate(input_train_list)

        normalized_output = 1
        if local_result < self.threshold:
            normalized_output = 0

        if normalized_output > expected_result:
            self.decrease_weights(input_train_list)
        elif normalized_output < expected_result:
            self.increase_weights(input_train_list)

    @staticmethod
    def activation_function(z):
        value = np.exp(-z)
        return 1. / (1. + value)
