import unittest
import backpropagation
import numpy as np


class Test1DNetwork(unittest.TestCase):
    """Testing the neural network structure where the network elements are lists of records
    (in contrary to single rows and multidimensional arrays)"""

    def setUp(self):
        """Set up testing objects"""
        inputs = [[0.99, 0.99, 0.01, 0.01], [0.99, 0.01, 0.99, 0.01]]
        results = [0.01, 0.99, 0.99, 0.01]

        # train the network
        self.net = backpropagation.Net("1dnet", inputs, results)

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
                [0.27259515, 0.31464586, 0.30016048, 0.29967086],
                [0.47259515, 0.50014794, 0.51588716, 0.49967086]
            ]
        )))
        self.assertTrue(np.allclose(self.net.getLayer(1).getNeuron(1).getWeights(), np.array(
            [
                [0.34799591, 0.4236796,  0.40025974, 0.39938152],
                [0.84799591, 0.90023919, 0.92571472, 0.89938152]
            ]
        )))
        self.assertTrue(np.allclose(self.net.getLayer(1).getNeuron(2).getWeights(), np.array(
            [
                [0.69452563, 0.84182632, 0.80045038, 0.79880217],
                [0.09452563, 0.20042249, 0.24458789, 0.19880217]
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
