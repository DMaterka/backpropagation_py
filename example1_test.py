import unittest
import backpropagation

class TestCalculator(unittest.TestCase):
    '''Testing the neural network structure'''

    def setUp(self):
        '''Set up testing objects'''
        self.a = 439
        self.b = 4

    def test_setting_input(self):
        '''Testing add menthod'''
        net = backpropagation.Net("1dimension", [[0.99, 0.99]], [[0.01]])
        # print(calculator.add())

        self.assertEqual(net.getInputs(), [[0.99, 0.99]])
        

if __name__ == '__main__':
    unittest.main(verbosity=2)