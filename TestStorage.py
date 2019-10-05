import unittest
import backpropagation
import train
import predict
import install
import sqlite3
import os
import dotenv


class TestStorage(unittest.TestCase):
    """Testing reading and writing network to db"""

    def setUp(self):
        os.environ["testing"] = "1"
        dotenv.load_dotenv('.env.testing')
        conn = sqlite3.connect('data/' + os.environ['DB_NAME'])
        install.createSchema(conn)
        conn.close()
        net = backpropagation.Net('xor.csv', 1)
        self.train_phase_output = train.train(net, "[3, 1]", 1000)
        print("result is", self.train_phase_output.getLayer(len(self.train_phase_output.getLayers()) - 1).getValues())
        net2 = backpropagation.Net('xor.csv', 1)
        self.predict_phase_output = predict.predict(net2)
        print("The result is ", self.predict_phase_output.getLayer(len(self.predict_phase_output.getLayers()) - 1).getValues())

    def test_stored_network_equals_read_one(self):
        """Testing set/get method"""
        self.assertAlmostEqual(
            self.train_phase_output.getLayer(len(self.train_phase_output.getLayers()) - 1).getValues().all(),
            self.predict_phase_output.getLayer(len(self.predict_phase_output.getLayers()) - 1).getValues().all()
        )
    
    def tearDown(self) -> None:
        os.remove('data/testing.db')

if __name__ == '__main__':
    unittest.main(verbosity=2)
