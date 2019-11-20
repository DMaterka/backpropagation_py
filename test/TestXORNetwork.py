import unittest
from src import backpropagation
import numpy as np


class TestXORNetwork(unittest.TestCase):
    """Testing the neural network structure where the network elements are lists of records
    (in contrary to single rows and multidimensional arrays)"""

    def setUp(self):
        # train the network
        self.net = backpropagation.Net("xor.csv")
        # set hidden layer
        hiddenLayer = backpropagation.Layer()
        hiddenLayer.setNeurons([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        hiddenLayer.setWeights([[.3, .5], [.4, .9], [.8, .2]])
        self.net.setLayer(1, hiddenLayer)

        # set output layer
        outputLayer = backpropagation.Layer()
        outputLayer.setNeurons([[0, 0, 0, 0]])
        self.net.setLayer(2, outputLayer)
        self.net.getLayer(2).setWeights([[.3, .5, .9]])

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
        self.assertTrue(
            np.allclose(self.net.getLayer(2).getNeuron(0).getValue(), [0.77807385, 0.74880408, 0.73841448, 0.70150502])
        )
        
    def test_backpropagation(self):
        """Testing network weights after backpropagation"""
        self.net.forwardPropagate()
        self.net.backPropagate()
        self.assertTrue(np.allclose(self.net.getLayer(1).getNeuron(0).getWeights(), np.array(
            [
                [0.26060975, 0.31347433, 0.30014579, 0.29956561],
                [0.46060975, 0.5001361,  0.51443298, 0.49956561]
            ]
        )))
        self.assertTrue(np.allclose(self.net.getLayer(1).getNeuron(1).getWeights(), np.array(
            [
                [0.33434959, 0.42245722, 0.40024298, 0.39927601],
                [0.83434959, 0.90022684, 0.92405497, 0.89927601]
        ]
        )))
        self.assertTrue(np.allclose(self.net.getLayer(1).getNeuron(2).getWeights(), np.array(
            [
                [0.68182925, 0.84042299, 0.80043736, 0.79869682],
                [0.08182925, 0.20040831, 0.24329894, 0.19869682]
            ]
        )))
        self.assertTrue(np.allclose(
            self.net.getLayer(2).getNeuron(0).getWeights(),
            np.array(
                [
                    [0.20871799, 0.32608355, 0.33022612, 0.22731131],
                    [0.39606824, 0.5272158,  0.53449924, 0.42713031],
                    [0.80330318, 0.93124455, 0.92679182, 0.82723891]
                ]
            )
        ))


if __name__ == '__main__':
    unittest.main(verbosity=2)
