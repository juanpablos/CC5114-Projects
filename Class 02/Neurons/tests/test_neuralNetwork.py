from unittest import TestCase

from Neurons.NeuralNetwork import NeuralNetwork
from Neurons.exceptions import LayerError


class TestNeuralNetwork(TestCase):
    network = None

    def setUp(self):
        self.network = NeuralNetwork()

    def setUpManual(self):
        self.network = NeuralNetwork(manual=True)

    def setUpNet(self):
        self.network = NeuralNetwork()
        self.network.add_layer(3)
        self.network.add_layer(4)
        self.network.initialize(2)

    def test_add_layer(self):
        self.assertEqual(len(self.network.layers), 0, "There should be no layer.")
        self.network.add_layer(10)
        self.assertEqual(len(self.network.layers), 1, "There should be 1 layer.")

    def test_initialize_exp(self):
        self.setUp()
        _pass = False
        try:
            self.network.initialize()
        except LayerError:
            _pass = True
        self.assertTrue(_pass, "It should raise a LayerError.")

    def test_initialize_manual_exp(self):
        self.setUpManual()
        _pass = False
        try:
            self.network.initialize()
        except LayerError:
            _pass = True
        self.assertTrue(_pass, "It should raise a LayerError.")

    def test_initialized(self):
        self.setUp()
        self.network.add_layer(4)
        self.network.add_layer(4)
        self.network.initialize(10)

        self.assertTrue(self.network.initialized, "It should be initialized.")

    def test_initialize(self):
        self.setUp()
        self.network.add_layer(4)
        self.network.add_layer(4)
        self.network.initialize(10)

        self.assertIs(self.network.firstLayer, self.network.layers[0], "The layers should be the same.")
        self.assertIs(self.network.lastLayer, self.network.layers[-1], "The layers should be the same.")

    def test_feed(self):
        self.setUpNet()
        out = self.network.feed([5, 10])
        self.assertEqual(len(out), 4, "The output length should be the same as neurons in the last layer.")
