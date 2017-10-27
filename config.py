# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
import numpy as np

class DbConfig:
	    DIR = 'data/'
	    NAME = 'backprop.db'

class Structure:
	    numit = 10000
	    dimensionsdef = [2,3,2,1]
	    inputs = np.array([[0,1,1],[0,0,0],[1,1,0],[1,0,1]])
	    out = 0
	    weights = {}
	    def __init__(self):
			   self.out = np.zeros((self.numit, len(self.inputs)))
			   #self.weights = np.random.rand(self.dimensionsdef)
			   self.weights[1] = np.array([[0.8, 0.2],[0.4, 0.9],[0.3, 0.5]])
			   self.weights[2] = np.array([[0.3, 0.76, 0.1],[0.5, 0.6, 0.4]])
			   self.weights[3] = np.array([[0.7], [0.3]]).transpose()