from unittest import TestCase

from Neurons.Neurons import Perceptron
from Neurons.exceptions import UnmatchedLengthError


class TestPerceptron(TestCase):
    perceptron = None

    def setUp(self):
        self.perceptron = Perceptron([2., 2.], -3., learning_rate=0.01)

    def test_init(self):
        self.setUp()
        self.assertEqual(self.perceptron.weights[0], 2., 'Should be 2')
        self.assertEqual(self.perceptron.weights[1], 2., 'Should be 2')

        self.assertEqual(self.perceptron.bias, -3., 'Should be -3')

    def test_evaluate(self):
        self.setUp()
        output = self.perceptron.evaluate([1., 1.])
        self.assertEqual(output, 1, 'Should be True-1')

    def test_evaluate2(self):
        self.setUp()
        with self.assertRaises(UnmatchedLengthError):
            self.perceptron.evaluate([1, 1, 4, 6, 8])

    def test_train(self):
        self.setUp()
        self.perceptron.train([1., 1.], 0)
        expected = 2 - self.perceptron.learning_rate * 1
        self.assertAlmostEqual(self.perceptron.weights[0], expected, delta=0.02)
        self.assertAlmostEqual(self.perceptron.weights[1], expected, delta=0.02)

    def test_train2(self):
        self.setUp()
        expected = 2 + self.perceptron.learning_rate * 0
        self.assertAlmostEqual(self.perceptron.weights[0], expected, delta=0.02)
        self.assertAlmostEqual(self.perceptron.weights[1], expected, delta=0.02)

    def test_decrease_weights(self):
        self.setUp()
        self.perceptron.decrease_weights([1., 1.])
        expected = 2 - self.perceptron.learning_rate * 1
        self.assertAlmostEqual(self.perceptron.weights[0], expected, delta=0.02)
        self.assertAlmostEqual(self.perceptron.weights[1], expected, delta=0.02)

    def test_increase_weights(self):
        self.setUp()
        self.perceptron.increase_weights([1., 1.])
        expected = 2 + self.perceptron.learning_rate * 1
        self.assertAlmostEqual(self.perceptron.weights[0], expected, delta=0.02)
        self.assertAlmostEqual(self.perceptron.weights[1], expected, delta=0.02)
