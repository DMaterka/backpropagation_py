import unittest
from src import backpropagation
import numpy as np


class TestNeuron(unittest.TestCase):

    def test_setValue(self):
        neuron = backpropagation.Neuron(True)
        self.assertRaises(Exception, neuron.setValue, 0.5)

    def test_getValue(self):
        neuron = backpropagation.Neuron(False)
        neuron.setValue(0.123)
        assert neuron.getValue() == 0.123
        neuron = backpropagation.Neuron(True)
        assert neuron.getValue() == 1

    def test_setSum(self):
        neuron = backpropagation.Neuron(True)
        self.assertRaises(Exception, neuron.setSum, 0.5)

    def test_getSum(self):
        neuron = backpropagation.Neuron(False)
        neuron.setSum(0.123)
        assert neuron.getSum() == 0.123
        neuron = backpropagation.Neuron(True)
        assert neuron.getSum() == 1

    def test_getDeltaSum(self):
        neuron = backpropagation.Neuron(False)
        neuron.setDeltaSum(0.123)
        assert neuron.getDeltaSum() == 0.123

    def test_getWeights(self):
        neuron = backpropagation.Neuron(False)
        neuron.setWeights([0.1, 0.2, 0.3])
        assert np.array_equal(neuron.getWeights(), [0.1, 0.2, 0.3])


if __name__ == '__main__':
    unittest.main(verbosity=2)
