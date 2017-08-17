from unittest import TestCase

from perceptron.NeuronLayer import NeuronLayer


class TestNeuronLayer(TestCase):
    layer = None

    def setUp(self):
        self.layer = NeuronLayer(4)

    def setUp2(self):
        self.layer = NeuronLayer(10)
        self.layer.initialize(2)

    def test_get_number_neurons(self):
        self.assertEqual(self.layer.get_number_neurons(), 4, 'Should be 4.')

    def test_pre_initialize(self):
        self.assertEqual(len(self.layer.neurons), 0, 'It should not be initialized.')

    def test_initialize(self):
        self.layer.initialize(2)
        self.assertEqual(len(self.layer.neurons), 4, 'There should be 4 initialized neurons.')

        self.assertEqual(len(self.layer.neurons[0].weights), 2, 'There should be 2 weight per neuron.')

    def test_evaluate_len(self):
        self.setUp2()
        output = self.layer.feed([1, 1])
        self.assertEqual(len(output), 10, 'There should be 10 outputs, as 10 neurons.')
