import unittest
import numpy as np
import train
import dotenv

class Test0DNetwork(unittest.TestCase):
    """Testing the neural network structure where the network elements are numbers
    (in contrary to lists and multidimensional arrays)"""

    def setUp(self):
        dotenv.load_dotenv('../.env.testing')
        inputfile = 'xor2.csv'
        learning_rate = 0.5
        structure = "[3]"

        training_sets = train.prepare_training_sets(inputfile)
        self.net = train.prepare_net(structure, learning_rate, training_sets, inputfile)
        self.net.getLayer(0).getNeuron(0).setValue(1)
        self.net.getLayer(0).getNeuron(1).setValue(0)
        self.net.getLayer(1).setWeights([[.3, .5], [.4, .9], [.8, .2]])
        self.net.getLayer(2).setWeights([[.3, .5, .9]])
        self.net.setExpectedResults([[0]])
        
    def test_forwardpropagation(self):
        """Testing network value after forwardpropagation"""
        self.net.forwardPropagate()
        # get network result
        value = self.net.getLayer(2).getNeuron(0).getValue()
        print(value)
        self.assertTrue(np.allclose(value, [0.77807385]))
        self.assertTrue(np.allclose(self.net.get_results(), [0.77807385]))
        
    def test_backpropagation(self):
        """TODO"""
        self.net.backPropagate()
        self.assertTrue(np.allclose(self.net.get_results(), [0.77807385]))

if __name__ == '__main__':
    unittest.main(verbosity=2)
