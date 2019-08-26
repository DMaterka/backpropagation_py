import unittest
import backpropagation
import numpy as np


class Test0DNetworkWithBias(unittest.TestCase):
    """Testing the neural network structure where the network elements are 1-dimensional arrays"""

    def setUp(self):
        """Set up testing objects"""
        test_inputs = [[.05], [.1]]
        test_outputs = [[.01], [.99]]

        # train the network
        self.net = backpropagation.Net("name", test_inputs, test_outputs, 0.5)

        # set input layer
        inputLayer = backpropagation.Layer()
        inputLayer.setNeurons(test_inputs, 1)
        inputLayer.setBias([.35, .35])
        self.net.setLayer(0, inputLayer)

        # set hidden layer
        hiddenLayer = backpropagation.Layer()
        hiddenLayer.setNeurons([[0], [0]])
        hiddenLayer.setBias([.6, .6])
        hiddenLayer.setWeights([[.15, .20], [.25, .3]])
        self.net.setLayer(1, hiddenLayer)

        # set output layer
        outputLayer = backpropagation.Layer()
        outputLayer.setNeurons([[0], [0]])
        outputLayer.setWeights([[.4, .45], [.50, .55]])
        self.net.setLayer(2, outputLayer)

    def test_weights(self):
        """Testing set/get method"""
        self.assertTrue(np.allclose(self.net.getLayer(1).getNeuron(0).getWeights(), np.array([.15, .20])))
        self.assertTrue(np.allclose(self.net.getLayer(1).getNeuron(1).getWeights(), np.array([.25, .3])))
        self.assertTrue(np.allclose(self.net.getLayer(2).getNeuron(0).getWeights(), np.array([.4, .45])))
        self.assertTrue(np.allclose(self.net.getLayer(2).getNeuron(1).getWeights(), np.array([.50, .55])))

    def test_forwardpropagation(self):
        """Testing network value after forwardpropagation"""
        self.net.forwardPropagate()
        # get network result
        self.assertTrue(np.allclose(self.net.getLayer(2).getNeuron(0).getValue(), [0.75136507]))
        self.assertTrue(np.allclose(self.net.getLayer(2).getNeuron(1).getValue(), [0.77292847]))
        
    def test_backpropagation(self):
        """Testing network weights after backpropagation"""
        self.net.forwardPropagate()
        self.net.backPropagate()
        self.assertTrue(np.allclose(self.net.getLayer(1).getNeuron(0).getWeights(), np.array([0.14924426, 0.19924426])))
        self.assertTrue(np.allclose(self.net.getLayer(1).getNeuron(1).getWeights(), np.array([0.24911876, 0.29911876])))
        self.assertTrue(np.allclose(self.net.getLayer(2).getNeuron(0).getWeights(), np.array([0.35891648, 0.40891648])))
        self.assertTrue(np.allclose(self.net.getLayer(2).getNeuron(1).getWeights(), np.array([0.51130127, 0.56130127])))

if __name__ == '__main__':
    unittest.main(verbosity=2)
