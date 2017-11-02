# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
import config as cfg
import numpy as np
import sqlite3
import random
import copy as cp

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
							 print("   I assign a value of " + str(value) + " to the neuron " + str(self) )
	    def getValue(self):
			   return self.value
	    def setSum(self, sum):
			   self.sum = sum
			   if debug:
							 print("   I assign a sum of " + str(sum) + " to the neuron " + str(self) )
	    def getSum(self):
			   return self.sum					 
class Layer:
	    #
	    # Get number of neurons
	    # set neurons' values 
	    # 
	    neurons = {}
	    layerSum = 0
	    def __init__(self):
			   pass
	    def setNeurons(self, sums, immutable = 0):
			   self.neurons = [Neuron() for i in range(len(sums))]
			   for i in range(len(self.neurons)):
					  if debug:
							 print("  I insert a " + str(i) + " neuron: " + str(self.neurons[i]) + " of layer " + str(self))
					  self.neurons[i].setSum(sums[i])
					  if immutable != 1:
							 self.neurons[i].setValue(ActivationFn().sigmoid(sums[i]))	
					  else:
							 self.neurons[i].setValue(sums[i])
	    def getNeurons(self):		   
			   return self.neurons
	    
	    def getValues(self):
			   tmparr = np.array([])
			   for i in (range(len(self.neurons))):
					  tmparr = np.append(tmparr,self.neurons[i].getValue())
			   return tmparr
	    def getSums(self):
			   tmparr = np.array([])
			   for i in (range(len(self.neurons))):
					  tmparr = np.append(tmparr,self.neurons[i].getSum())
			   return tmparr
class Net:
	    # contains layers
	    # make a forward and backpropagation
	    # return layer at any moment
	    dims = 0
	    name = ''	 
	    structure = []
	    weights = []
	    layers = {}
	    deltaOutputSum = 0 
	    error = 0
	    
	    def __init__(self,structure,name):
			   self.structure = cfg.Structure()
			   self.weights = self.structure.weights
			   self.setName(name)
			   for i in range(1,self.structure.numit):
					  self.forwardPropagate()
					  self.backPropagate()
					  
	    def setDimensionNumber(self, dims):
			   self.dims = dims
	    def getDimensionNumber(self):	   
			   return self.dims
	    def setName(self, name):
			   self.name = name
	    def getName(self):	   
			   return self.name
	    def forwardPropagate(self):
			   self.layers[0] = Layer() 
			   if debug:
							 print("I set the " + str(0) + " layer: " + str(self.layers[0]) + " of network " + str(self))
			   self.layers[0].setNeurons([1,1],1)
			   
			   for i in range(1,len(self.structure.dimensionsdef)-1):
					  self.layers[i] = Layer()
					  if debug:
							 print("I set the " + str(i) + " layer: " + str(self.layers[i-1]) + " of network " + str(self))
					  #produce neurons' sums		
					  values = np.dot(self.structure.weights[i], self.layers[i-1].getValues())
					  self.layers[i].setNeurons(values)
			   			 
			   self.layers[i+1] = Layer()
			   if debug:
							 print("I set the " + str(i+1) + " layer: " + str(self.layers[i]) + " of network " + str(self))
			   value = np.dot(self.structure.weights[len(self.structure.weights)], self.layers[i].getValues())
			   self.layers[i+1].setNeurons(value)
			   self.error = 0 - self.layers[i+1].getNeurons()[0].getValue()
			   
	    def backPropagate(self):
			   oldweights = cp.copy(self.weights)
			   i = len(self.weights)
			   deltaSum = ActivationFn().sigmoidprime(self.layers[i].getSums()[0]) * self.error
			   deltaWeights = deltaSum / self.layers[i-1].getValues()
			   self.weights[i] = self.weights[i] + deltaWeights
			   
			   for j in range(len(self.weights)-1,0,-1):
					  deltaSum = deltaSum / oldweights[j+1] * ActivationFn().sigmoidprime(self.layers[(j)].getSums())
					  #allow legally divise the arrays with different shapes, use instead for over all inputs
					  deltaWeigths = np.tile(deltaSum,(np.size(self.layers[(j-1)].getValues()),1)).T / self.layers[(j-1)].getValues().reshape(1,2)
					  self.weights[j] = self.weights[j] + deltaWeigths
					  oldweights = cp.copy(self.weights)
			   print(self.weights)
class ActivationFn():
	    @staticmethod
	    def sigmoid(x):
			   x = np.array(x)
			   return 1/(1+np.exp(-x))
	    
	    @staticmethod
	    def sigmoidprime(x):
			   return (__class__.sigmoid(x))*(1 - __class__.sigmoid(x))
			   
#test = np.array([[-0.03325106, -0.13300425],[-0.06650212, -0.0295565 ],[-0.0532017, -0.0886695 ]])
#print(test)
#print(test.T)
#print(test / np.array([1, 1]) )

net = Net(cfg.Structure,"name")

#net.forwardPropagate();

conn = sqlite3.connect(cfg.DbConfig.DIR + cfg.DbConfig.NAME)

conn.close