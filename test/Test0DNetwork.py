import unittest
from src import backpropagation
import numpy as np


class Test0DNetwork(unittest.TestCase):
    """Testing the neural network structure where the network elements are numbers
    (in contrary to lists and multidimensional arrays)"""

    def setUp(self):
        """Set up testing objects"""
        # train the network
        self.net = backpropagation.Net("xor1d.csv", 1)
        # set hidden layer
        hiddenLayer = backpropagation.Layer()
        hiddenLayer.setNeurons([[0], [0], [0]])
        hiddenLayer.setWeights([[.3, .5], [.4, .9], [.8, .2]])
        self.net.setLayer(1, hiddenLayer)

        # set second hidden layer
        outputLayer = backpropagation.Layer()
        outputLayer.setNeurons([[0]])
        self.net.setLayer(2, outputLayer)
        self.net.getLayer(2).setWeights([[.3, .5, .9]])
        self.net.setExpectedResults([[0]])
        
    def test_weights(self):
        """Testing set/get method"""
        self.assertTrue(np.allclose(self.net.getLayer(1).getNeuron(0).getWeights(), np.array([.3, .5])))
        self.assertTrue(np.allclose(self.net.getLayer(1).getNeuron(1).getWeights(), np.array([.4, .9])))
        self.assertTrue(np.allclose(self.net.getLayer(1).getNeuron(2).getWeights(), np.array([.8, .2])))
        self.assertTrue(np.allclose(self.net.getLayer(2).getNeuron(0).getWeights(), np.array([.3, .5, .9])))

    def test_forwardpropagation(self):
        """Testing network value after forwardpropagation"""
        self.net.forwardPropagate()
        # get network result
        value = self.net.getLayer(2).getNeuron(0).getValue()
        self.assertTrue(value, [0.77807385])
        
    def test_backpropagation(self):
        """Testing network weights after backpropagation"""
        self.net.forwardPropagate()
        self.net.backPropagate()
        self.assertTrue(np.allclose(self.net.getLayer(1).getNeuron(0).getWeights(), np.array([[0.26060975], [0.46060975]])))
        self.assertTrue(np.allclose(self.net.getLayer(1).getNeuron(1).getWeights(), np.array([[0.33434959], [0.83434959]])))
        self.assertTrue(np.allclose(self.net.getLayer(1).getNeuron(2).getWeights(), np.array([[0.68182925], [0.08182925]])))
        self.assertTrue(np.allclose(
            self.net.getLayer(2).getNeuron(0).getWeights(),
            np.array([[0.20871799], [0.39606824], [0.80330318]])
        ))


if __name__ == '__main__':
    unittest.main(verbosity=2)
