import unittest

import dotenv
import src.dbops


class Predict(unittest.TestCase):

    def setUp(self) -> None:
        dotenv.load_dotenv('../../.env.testing')
        self.net = src.dbops.DbOps().load_net('noisy_xor.csv')

    def test_xor1(self):
        values = [0.75, -0.25]
        self.net.setInputs(values)
        neurons = self.net.getLayer(0).getNeurons()
        for ind, neuron in enumerate(neurons):
            neuron.setValue(values[ind])
            neuron.setSum(values[ind])
        self.net.forwardPropagate()
        self.assertGreater(self.net.get_results()[0], 0.5)

    def test_xor2(self):
        values = [0.75, 0.75]
        self.net.setInputs(values)
        neurons = self.net.getLayer(0).getNeurons()
        for ind, neuron in enumerate(neurons):
            neuron.setValue(values[ind])
            neuron.setSum(values[ind])
        self.net.forwardPropagate()
        self.assertLess(self.net.get_results()[0], 0.5)

    def test_xor3(self):
        values = [-0.25, 0.75]
        self.net.setInputs(values)
        neurons = self.net.getLayer(0).getNeurons()
        for ind, neuron in enumerate(neurons):
            neuron.setValue(values[ind])
            neuron.setSum(values[ind])
        self.net.forwardPropagate()
        self.assertGreater(self.net.get_results()[0], 0.5)

    def test_xor4(self):
        values = [-0.25, -0.25]
        self.net.setInputs(values)
        neurons = self.net.getLayer(0).getNeurons()
        for ind, neuron in enumerate(neurons):
            neuron.setValue(values[ind])
            neuron.setSum(values[ind])
        self.net.forwardPropagate()
        self.assertLess(self.net.get_results()[0], 0.5)

if __name__ == '__main__':
    unittest.main()
