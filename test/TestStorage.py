import unittest
import train
from src import dbops
import os
import dotenv


class TestStorage(unittest.TestCase):
    """Testing reading and writing network to db"""

    def setUp(self):
        dotenv.load_dotenv('../.env.testing')
        inputfile = 'xor.csv'
        learning_rate = 0.5
        structure = "[2]"
        self.model_name = 'storage_test_model'
        training_sets = train.prepare_training_sets(inputfile)
        self.net = train.prepare_net(structure, learning_rate, training_sets, inputfile)
    
        self.net.getLayer(0).setBias([.35, .35])
        self.net.getLayer(1).setWeights([[.15, .2], [.25, .3]]).setBias([.6, .6])
        self.net.getLayer(2).setWeights([[.4, .45], [.5, .55]])
        if not os.path.isfile(os.environ['PROJECT_ROOT'] + 'test/data/' + os.environ['DB_NAME']):
            dbops.createSchema(os.environ['DB_NAME'])
        dbops.save_net(self.net, 100, self.model_name)

    def test_stored_network_equals_read_one(self):
        loaded_net = dbops.load_net(self.model_name)
        # test if first layer bias are the same
        self.assertAlmostEqual(self.net.getLayer(0).getBias(), loaded_net.getLayer(0).getBias())
    
    def tearDown(self) -> None:
        dbops.delete_model(self.model_name)

if __name__ == '__main__':
    unittest.main(verbosity=2)
