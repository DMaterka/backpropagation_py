import unittest
import train
from src import dbops
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
        dbOpsObject = dbops.DbOps()
        self.net.getLayer(0).setBias([.35, .35])
        self.net.getLayer(1).setWeights([[.15, .2], [.25, .3]]).setBias([.6, .6])
        self.net.getLayer(2).setWeights([[.4, .45], [.5, .55]])
        dbOpsObject.save_net(self.net, 100, self.model_name)

    def test_stored_network_equals_read_one(self):
        dbOpsObject = dbops.DbOps()
        loaded_net = dbOpsObject.load_net(self.model_name)
        # test if first layer bias are the same
        first_net_first_layer_weights = self.net.getLayer(0).getBias().getWeights()
        second_net_first_layer_weights = loaded_net.getLayer(0).getBias().getWeights()
        self.assertEqual(first_net_first_layer_weights.any(), second_net_first_layer_weights.any())
    
    def tearDown(self) -> None:
        dbOpsObject = dbops.DbOps()
        dbOpsObject.delete_model(self.model_name)

if __name__ == '__main__':
    unittest.main(verbosity=2)
