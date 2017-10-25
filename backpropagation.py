# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
import config as cfg
import numpy as np
import sqlite3
import random

debug = 1

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
							 print("I assign a value " + str(value) + " of neuron " + str(self) )
	    def getValue(self):
			   return self.value
class Layer:
	    #
	    # Get number of neurons
	    # set neurons' values 
	    # 
	    neurons = {}
	    layerSum = 0
	    def __init__(self):
			   pass
	    def setNeurons(self, values):
			   self.neurons = [Neuron() for i in range(len(values))]
			   lsum = 0
			   for i in range(len(self.neurons)):
					  if debug:
							 print("I insert a " + str(i) + " neuron: " + str(self.neurons[i]) + " of layer " + str(self))
					  self.neurons[i].setValue(values[i])	
	    def getNeurons(self):		   
			   return self.neurons
	    
	    def getValues(self):
			   tmparr = []
			   for i in (range(len(self.neurons))):
					  tmparr.append(self.neurons[i].getValue())
			   return tmparr		
class Net:
	    # contains layers
	    # make a forward and backpropagation
	    # return layer at any moment
	    dims = 0
	    name = ''	 
	    structure = []
	    layers = {}
	    
	    def __init__(self,structure,name):
			   self.structure = cfg.Structure()
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
			   
			   self.layers = [Layer() for i in range(0,len(self.structure.dimensionsdef))]
			   if debug:
							 print("I set the " + str(0) + " layer: " + str(self.layers[0]) + " of network " + str(self))
			   self.layers[0].setNeurons([1,1])
			   
			   for i in range(1,len(self.layers)):
					  #self.layers[i] = Layer(str(i))
					  if debug:
							 print("I set the " + str(i) + " layer: " + str(self.layers[i]) + " of network " + str(self))
							 
					  values = np.dot(self.structure.weights[i], self.layers[i-1].getValues())
					  self.layers[i].setNeurons(values)
	    def forwardPropagate(self):
			   pass
#	    def backPropagate(self):
#			   pass
					  
class ActivationFn():
	    @staticmethod
	    def sigmoid(x):
			   return 1/(1+np.exp(-x))
	    
	    @staticmethod
	    def sigmoidprime(x):
			   return (__class__.sigmoid(x))*(1-__class__.sigmoid(x))
			   
net = Net(cfg.Structure,"name")

net.forwardPropagate();

conn = sqlite3.connect(cfg.DbConfig.DIR + cfg.DbConfig.NAME)

conn.close