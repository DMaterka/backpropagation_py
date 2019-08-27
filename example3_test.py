import unittest
import backpropagation
import numpy as np


class Test0DNetworkWithMultipleHiddenLayers(unittest.TestCase):
    """Testing the neural network structure where the network elements are 1-dimensional arrays
        there are multiple layers """

    def setUp(self):
        """Set up testing objects"""
        inputs = [[.01], [.99], [.01], [.01], [.99], [.99], [.99], [.01]]
        results = [[.99], [.01], [.01], [.99]]

        # train the network
        self.net = backpropagation.Net("name", inputs, results, 0.3)
        # set input layer
        inputLayer = backpropagation.Layer()
        inputLayer.setNeurons(inputs, 1)
        inputLayer.setBias([.4, .6])
        self.net.setLayer(0, inputLayer)
        # set hidden layer
        hiddenLayer = backpropagation.Layer()
        hiddenLayer.setNeurons([[0], [0]])
        hiddenLayer.setBias([.4, .6, .5])
        hiddenLayer.setWeights([[.15, .2, .1, .6, .2, .9, .5, .3], [.15, .2, .1, .6, .2, .9, .5, .3]])
        self.net.setLayer(1, hiddenLayer)

        # set second hidden layer
        hiddenLayer1 = backpropagation.Layer()
        hiddenLayer1.setNeurons([[0], [0], [0]])
        hiddenLayer1.setBias([.2, .7, .9, .3, .8])
        hiddenLayer1.setWeights([[.4, .55], [.5, .55], [.5, .55]])
        self.net.setLayer(2, hiddenLayer1)

        hiddenLayer2 = backpropagation.Layer()
        hiddenLayer2.setNeurons([[0], [0], [0], [0], [0]])
        hiddenLayer2.setBias([.5, .2, .6, .8])
        hiddenLayer2.setWeights([[.6, .8, .3], [.6, .8, 0.3], [.6, .8, .3], [.6, .8, .3], [.6, .8, .3]])
        self.net.setLayer(3, hiddenLayer2)

        # set output layer
        outputLayer = backpropagation.Layer()
        outputLayer.setNeurons([[0], [0], [0], [0]])
        outputLayer.setWeights([[.3, .5, .9, .7, .5], [.3, .5, .9, .7, .5], [.3, .5, .9, .7, .5], [.3, .5, .9, .7, .5]])
        self.net.setLayer(4, outputLayer)

    def test_weights(self):
        """Testing set/get method"""
        self.assertTrue(np.allclose(self.net.getLayer(1).getNeuron(0).getWeights(), np.array([.15, .2, .1, .6, .2, .9, .5, .3])))
        self.assertTrue(np.allclose(self.net.getLayer(1).getNeuron(1).getWeights(), np.array([.15, .2, .1, .6, .2, .9, .5, .3])))
        self.assertTrue(np.allclose(self.net.getLayer(2).getNeuron(0).getWeights(), np.array([.4, .55])))
        self.assertTrue(np.allclose(self.net.getLayer(2).getNeuron(1).getWeights(), np.array([.5, .55])))
        self.assertTrue(np.allclose(self.net.getLayer(2).getNeuron(2).getWeights(), np.array([.5, .55])))
        self.assertTrue(np.allclose(self.net.getLayer(3).getNeuron(0).getWeights(), np.array([.6, .8, .3])))
        self.assertTrue(np.allclose(self.net.getLayer(3).getNeuron(1).getWeights(), np.array([.6, .8, .3])))
        self.assertTrue(np.allclose(self.net.getLayer(3).getNeuron(2).getWeights(), np.array([.6, .8, .3])))
        self.assertTrue(np.allclose(self.net.getLayer(3).getNeuron(3).getWeights(), np.array([.6, .8, .3])))
        self.assertTrue(np.allclose(self.net.getLayer(3).getNeuron(4).getWeights(), np.array([.6, .8, .3])))
        self.assertTrue(np.allclose(self.net.getLayer(4).getNeuron(0).getWeights(), np.array([.3, .5, .9, .7, .5])))
        self.assertTrue(np.allclose(self.net.getLayer(4).getNeuron(1).getWeights(), np.array([.3, .5, .9, .7, .5])))
        self.assertTrue(np.allclose(self.net.getLayer(4).getNeuron(2).getWeights(), np.array([.3, .5, .9, .7, .5])))
        self.assertTrue(np.allclose(self.net.getLayer(4).getNeuron(3).getWeights(), np.array([.3, .5, .9, .7, .5])))

    def test_forwardpropagation(self):
        """Testing network value after forwardpropagation"""
        self.net.forwardPropagate()
        # get network result
        self.assertTrue(np.allclose(self.net.getLayer(1).getNeuron(0).getValue(), [0.89966429]))
        self.assertTrue(np.allclose(self.net.getLayer(1).getNeuron(1).getValue(), [0.9163303]))
        self.assertTrue(np.allclose(self.net.getLayer(2).getNeuron(0).getValue(), [0.7796877]))
        self.assertTrue(np.allclose(self.net.getLayer(2).getNeuron(1).getValue(), [0.82546388]))
        self.assertTrue(np.allclose(self.net.getLayer(2).getNeuron(2).getValue(), [0.81058469]))
        self.assertTrue(np.allclose(self.net.getLayer(3).getNeuron(0).getValue(), [0.82797728]))
        self.assertTrue(np.allclose(self.net.getLayer(3).getNeuron(1).getValue(), [0.88808811]))
        self.assertTrue(np.allclose(self.net.getLayer(3).getNeuron(2).getValue(), [0.90647707]))
        self.assertTrue(np.allclose(self.net.getLayer(3).getNeuron(3).getValue(), [0.84175694]))
        self.assertTrue(np.allclose(self.net.getLayer(3).getNeuron(4).getValue(), [0.89764791]))
        
    def test_backpropagation(self):
        """Testing network weights after backpropagation"""
        self.net.forwardPropagate()
        self.net.backPropagate()
        self.assertTrue(np.allclose(self.net.getLayer(1).getNeuron(0).getWeights(), np.array(
            [0.14947499, 0.19947499, 0.09947499, 0.59947499, 0.19947499, 0.89947499, 0.49947499, 0.29947499]
        )))
        self.assertTrue(np.allclose(self.net.getLayer(1).getNeuron(1).getWeights(), np.array(
            [0.14936888, 0.19936888, 0.09936888, 0.59936888, 0.19936888, 0.89936888, 0.49936888, 0.29936888]
        )))
        self.assertTrue(np.allclose(self.net.getLayer(2).getNeuron(0).getWeights(), np.array(
            [0.36046602, 0.51046602]
        )))
        self.assertTrue(np.allclose(self.net.getLayer(2).getNeuron(1).getWeights(), np.array(
            [0.446992, 0.496992]
        )))
        self.assertTrue(np.allclose(self.net.getLayer(2).getNeuron(2).getWeights(), np.array(
            [0.48067704, 0.53067704]
        )))
        self.assertTrue(np.allclose(self.net.getLayer(3).getNeuron(0).getWeights(), np.array(
            [0.59407747, 0.79407747, 0.29407747]
        )))
        self.assertTrue(np.allclose(self.net.getLayer(3).getNeuron(1).getWeights(), np.array(
            [0.58996702, 0.78996702, 0.28996702]
        )))
        self.assertTrue(np.allclose(self.net.getLayer(3).getNeuron(2).getWeights(), np.array(
            [0.58174612, 0.78174612, 0.28174612]
        )))
        self.assertTrue(np.allclose(self.net.getLayer(3).getNeuron(3).getWeights(), np.array(
            [0.58585657, 0.78585657, 0.28585657]
        )))
        self.assertTrue(np.allclose(self.net.getLayer(3).getNeuron(4).getWeights(), np.array(
            [0.58996702, 0.78996702, 0.28996702]
        )))
        self.assertTrue(np.allclose(self.net.getLayer(4).getNeuron(0).getWeights(), np.array(
            [0.30038064, 0.50038064, 0.90038064, 0.70038064, 0.50038064]
        )))
        self.assertTrue(np.allclose(self.net.getLayer(4).getNeuron(1).getWeights(), np.array(
            [0.28691552, 0.48691552, 0.88691552, 0.68691552, 0.48691552]
        )))
        self.assertTrue(np.allclose(self.net.getLayer(4).getNeuron(2).getWeights(), np.array(
            [0.29068268, 0.49068268, 0.89068268, 0.69068268, 0.49068268]
        )))
        self.assertTrue(np.allclose(self.net.getLayer(4).getNeuron(3).getWeights(), np.array(
            [0.30019602, 0.50019602, 0.90019602, 0.70019602, 0.50019602]
        )))

if __name__ == '__main__':
    unittest.main(verbosity=2)
