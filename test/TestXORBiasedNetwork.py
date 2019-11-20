import unittest
from src import backpropagation
import numpy as np


class TestXORBiasedNetwork(unittest.TestCase):
    """Testing the neural network structure where the network elements are lists of records
    (in contrary to single rows and multidimensional arrays)"""

    def setUp(self):
        # train the network
        self.net = backpropagation.Net("xor.csv")
        # set hidden layer
        hiddenLayer = backpropagation.Layer()
        hiddenLayer.setNeurons([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        hiddenLayer.setWeights([[.3, .5], [.4, .9], [.8, .2]])
        hiddenLayer.setBias([[.3, .4, .8]])
        self.net.setLayer(1, hiddenLayer)

        # set output layer
        outputLayer = backpropagation.Layer()
        outputLayer.setNeurons([[0, 0, 0, 0]])
        outputLayer.setWeights([[.3, .5, .9]])
        outputLayer.setBias([.2])
        self.net.setLayer(2, outputLayer)

    def test_biases(self):
        """Testing set/get method"""
        self.assertTrue(np.allclose(self.net.getLayer(1).getBias().getWeights(), np.array([.3, .4, .8])))
        self.assertTrue(np.allclose(self.net.getLayer(2).getBias().getWeights(), np.array([.2])))

    def test_forwardpropagation(self):
        """Testing network value after forwardpropagation"""
        self.net.forwardPropagate()
        # get network result
        self.assertTrue(
            np.allclose(self.net.getLayer(2).getNeuron(0).getValue(), [0.82555938, 0.80095027, 0.7921187, 0.76032734])
        )
        
    def test_backpropagation(self):
        """Testing network weights after backpropagation"""
        self.net.forwardPropagate()
        self.net.backPropagate()
        self.assertTrue(np.allclose(self.net.getLayer(1).getNeuron(0).getWeights(), np.array(
            [
                [.26511747, .30895158, .30009775, .2995898],
                [.46511747, .50009042, .50967758, .4995898]
        ]
        )))
        self.assertTrue(np.allclose(self.net.getLayer(1).getNeuron(1).getWeights(), np.array(
            [
                [.34186245, .4149193, .40016292, .39931634],
                [.84186245, .9001507, .9161293, .89931634]
            ]
        )))
        self.assertTrue(np.allclose(self.net.getLayer(1).getNeuron(2).getWeights(), np.array(
            [
                [.69535241, .82685474, .80029326, .79876941],
                [.09535241, .20027126, .22903275, .19876941]
            ]
        )))
        print(self.net.getLayer(2).getNeuron(0).getWeights())
        self.assertTrue(np.allclose(
            self.net.getLayer(2).getNeuron(0).getWeights(),
            np.array(
                [
                    [.21916407, .31732843, .32026718, .23136058],
                    [.40796192, .51808063, .52313239, .43118967],
                    [.81436892, .9207571, .91796442, .83129222]
                ]
            )
        ))


if __name__ == '__main__':
    unittest.main(verbosity=2)
