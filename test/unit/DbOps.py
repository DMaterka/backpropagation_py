import unittest
import train
from src import dbops
import dotenv
import numpy.testing as np_test


class TestStorage(unittest.TestCase):
    """Testing reading and writing network to db"""

    def setUp(self):
        dotenv.load_dotenv('../../.env.testing')
        inputfile = 'mmazur_example.csv'
        self.model_name = 'storage_test_model'
        training_sets = train.prepare_training_sets(inputfile)

        self.net = train.prepare_net(
            hidden_structure="[2]",
            learning_rate=0.5,
            training_sets=training_sets,
            inputfile=inputfile
        )

        dbOpsObject = dbops.DbOps()
        self.net.getLayer(0).setBias([.35, .35])
        self.net.getLayer(1).setWeights([[.15, .2], [.25, .3]]).setBias([.6, .6])
        self.net.getLayer(2).setWeights([[.4, .45], [.5, .55]])
        dbOpsObject.save_net(self.net, 100, self.model_name)

        dbOpsObject = dbops.DbOps()
        self.loaded_net = dbOpsObject.load_net(self.model_name)

    def test_first_layer_bias_equals_read_one(self):
        np_test.assert_allclose(
            self.net.getLayer(0).getBias().getWeights(),
            self.loaded_net.getLayer(0).getBias().getWeights()
        )

    def test_first_layer_values_equals_read_one(self):
        np_test.assert_allclose(
            self.net.getLayer(0).getValues(),
            self.loaded_net.getLayer(0).getValues()
        )
    
    def test_first_layer_sums_equals_read_one(self):
        np_test.assert_allclose(
            self.net.getLayer(0).getSums(), self.loaded_net.getLayer(0).getSums()
        )

    def test_second_layer_bias_equals_read_one(self):
        # test if first layer bias are the same
        np_test.assert_allclose(
            self.net.getLayer(1).getBias().getWeights(),
            self.loaded_net.getLayer(1).getBias().getWeights()
        )
        
    def test_second_layer_weights_equals_read_one(self):
        np_test.assert_allclose(self.net.getLayer(1).getWeights(), self.loaded_net.getLayer(1).getWeights())
    
    def test_second_layer_sums_equals_read_one(self):
        np_test.assert_allclose(self.net.getLayer(1).getSums(), self.loaded_net.getLayer(1).getSums())
        
    def test_second_layer_values_equals_read_one(self):
        np_test.assert_allclose(self.net.getLayer(1).getValues(), self.loaded_net.getLayer(1).getValues())
        
    def test_third_layer_weights_equals_read_one(self):
        np_test.assert_allclose(self.net.getLayer(2).getWeights(), self.loaded_net.getLayer(2).getWeights())

    def test_third_layer_sums_equals_read_one(self):
        np_test.assert_allclose(self.net.getLayer(2).getSums(), self.loaded_net.getLayer(2).getSums())

    def test_third_layer_values_equals_read_one(self):
        np_test.assert_allclose(self.net.getLayer(2).getValues(), self.loaded_net.getLayer(2).getValues())
    
    def tearDown(self) -> None:
        dbOpsObject = dbops.DbOps()
        dbOpsObject.delete_model(self.model_name)


if __name__ == '__main__':
    unittest.main(verbosity=2)
