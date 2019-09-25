import unittest
import backpropagation
import train
import predict
import pandas as pd


class TestStorage(unittest.TestCase):
    """Testing reading and writing network to db"""

    def setUp(self):
        df = pd.read_csv('xor.csv')
        net = backpropagation.Net('xor.csv', [df["col1"], df["col2"]], [df["exp_result"]], 1)
        self.train_phase_output = train.train(net, "[3, 1]", 1000)
        print("result is", self.train_phase_output.getLayer(len(self.train_phase_output.getLayers()) - 1).getValues())
        net2 = backpropagation.Net('xor.csv', [df["col1"], df["col2"]], [df["exp_result"]], 1)
        self.predict_phase_output = predict.predict(net2)
        print("The result is ", self.predict_phase_output.getLayer(len(self.predict_phase_output.getLayers()) - 1).getValues())

    def test_stored_network_equals_read_one(self):
        """Testing set/get method"""
        self.assertEqual(self.train_phase_output, self.predict_phase_output)


if __name__ == '__main__':
    unittest.main(verbosity=2)
