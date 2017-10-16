# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
import config as cfg
import numpy as np
import sqlite3
import random

debug = 0

class Neuron:
	    #basic type
	    #set value
	    #set sum
	    
	    value = 0
	    sum = 0
	    def __init__(self):
			   pass
	    def setValue(self, value):
			   self.value = value
			   if debug:
							 print("I set a value of " + str(value) + " of neuron " + str(self) )
	    def getValue(self):
			   return self.value
class Layer:
	    #
	    # Get number of neurons
	    # set neurons' values 
	    # 
	    neurons = {}
	    def __init__(self):
			   pass
	    def setNeurons(self, length):
			   self.neurons = [Neuron() for i in range(length)]
			   for i in range(length):
					  if debug:
							 print("I insert a " + str(i) + " neuron: " + str(self.neurons[i]) + " of layer " + str(self))
					  self.neurons[i].setValue(random.random())		 
	    def getNeurons(self,id):		   
			   return self.neurons[id]

class Net:
	    # contains layers
	    # make a forward and backpropagation
	    # return layer at any moment
	    dims = 0
	    name = ''	 
	    structure = []
	    layers = {}
	    
	    def __init__(self,structure,name):
			   self.structure = structure
			   self.setName(name)
			   self.buildArray()
	    def setDimensionNumber(self, dims):
			   self.dims = dims
	    def getDimensionNumber(self):	   
			   return self.dims
	    def setName(self, name):
			   self.name = name
	    def getName(self):	   
			   return self.name
	    def buildArray(self):
			   self.layers = [Layer() for i in range(len(self.structure.dimensionsdef))]
			   for i in range(len(self.layers)):
					  #self.layers[i] = Layer(str(i))
					  if debug:
							 print("I set the " + str(i) + " layer: " + str(self.layers[i]) + " of network " + str(self))
					  layerLength = self.structure.dimensionsdef[i]
					  self.layers[i].setNeurons(layerLength)
	    def forwardPropagate(self):
			   pass
	    def backPropagate(self):
			   pass
					  
class ActivationFn():
	    @staticmethod
	    def sigmoid(x):
			   return 1/(1+np.exp(-x))
	    
	    @staticmethod
	    def sigmoidprime(x):
			   return (__class__.sigmoid(x))*(1-__class__.sigmoid(x))
			   
net = Net(cfg.Structure,"name")

conn = sqlite3.connect(cfg.DbConfig.DIR + cfg.DbConfig.NAME)

conn.close