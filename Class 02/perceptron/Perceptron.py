import numpy as np

learningRate = 0.01

class Perceptron:

    def __init__(self, weightList, bias):
        self.weights = np.array(weightList)
        self.bias = bias

    def evaluate(self, inputList):
        try:
            assert len(self.weights) == len(inputList)

            resSum = sum(self.weights * np.array(inputList))
            res = 0
            if resSum + self.bias > 0:
                res = 1
            return res

        except AssertionError:
            print("The length of the input defers from the weights vector.")
        except TypeError:
            print("You either entered a non number list in constructor or input. Types don't match.")

    def train(self, trainInputList, expectedResult):
        localResult = self.evaluate(trainInputList)
        if localResult > expectedResult:
            self.decreaseWeights(trainInputList)
        elif localResult < expectedResult:
            self.increaseWeights(trainInputList)


    def decreaseWeights(self, inputs):
        self.weights -= np.array(inputs) * learningRate

    def increaseWeights(self, inputs):
        self.weights += np.array(inputs) * learningRate
