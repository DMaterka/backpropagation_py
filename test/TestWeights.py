import unittest
import numpy as np
import train
import dotenv


class TestWeights(unittest.TestCase):
    """Testing the neural network structure where the network elements are numbers
    (in contrary to lists and multidimensional arrays)"""

    def setUp(self):
        dotenv.load_dotenv('../.env.testing')
        inputfile = 'xor2.csv'
        learning_rate = 0.5
        structure = "[3]"

        training_sets = train.prepare_training_sets(inputfile)
        self.net = train.prepare_net(structure, learning_rate, training_sets, inputfile)
        self.net.getLayer(1).setWeights([[.3, .5], [.4, .9], [.8, .2]])
        self.net.getLayer(2).setWeights([[.3, .5, .9]])
        self.net.setExpectedResults([[0]])

    def test_get_weights(self):
        self.assertTrue(np.allclose(self.net.getLayer(1).getNeuron(0).getWeights(), np.array([.3, .5])))
        self.assertTrue(np.allclose(self.net.getLayer(1).getNeuron(1).getWeights(), np.array([.4, .9])))
        self.assertTrue(np.allclose(self.net.getLayer(1).getNeuron(2).getWeights(), np.array([.8, .2])))
        self.assertTrue(np.allclose(self.net.getLayer(2).getNeuron(0).getWeights(), np.array([.3, .5, .9])))

    def test_get_weights_after_backpropagation(self):
        self.net.backPropagate()
        self.assertTrue(
            np.allclose(self.net.getLayer(1).getNeuron(0).getWeights(), np.array([[0.26060975], [0.46060975]]))
        )
        self.assertTrue(
            np.allclose(self.net.getLayer(1).getNeuron(1).getWeights(), np.array([[0.33434959], [0.83434959]]))
        )
        self.assertTrue(
            np.allclose(self.net.getLayer(1).getNeuron(2).getWeights(), np.array([[0.68182925], [0.08182925]]))
        )
        self.assertTrue(
            np.allclose(
                self.net.getLayer(2).getNeuron(0).getWeights(),
                np.array([[0.20871799], [0.39606824], [0.80330318]])
            )
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)
