# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
import numpy as np

class DbConfig:
	    DIR = 'data/'
	    NAME = 'backprop.db'

class Structure:
	    numit = 10000
	    dimensionsdef = [2,3,1]
	    inputs = np.array([[0, 0, 1, 1], [1, 0, 1, 0]])
	    expected = [1, 0, 0, 1]
	    out = 0
	    def __init__(self):
			   self.out = np.zeros((self.numit, len(self.inputs)))