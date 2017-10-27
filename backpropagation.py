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
	    deltaOutputSum = 0 
	    error = 0
	    
	    def __init__(self,structure,name):
			   self.structure = cfg.Structure()
			   self.setName(name)
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
			   self.error = abs(0 - value)
			   
	    def backPropagate(self):
			   deltaOutputSum = ActivationFn().sigmoidprime(self.layers[len(self.layers)-1].getValues()[0]) * self.error
			   print(deltaOutputSum)
			   weightsDiff = deltaOutputSum / self.structure.weights[len(self.structure.weights)]
			   print(weightsDiff)
			   self.structure.weights[len(self.structure.weights)] = self.structure.weights[len(self.structure.weights)] + weightsDiff
			   print(self.structure.weights[len(self.structure.weights)])
					  
class ActivationFn():
	    @staticmethod
	    def sigmoid(x):
			   return 1/(1+np.exp(-x))
	    
	    @staticmethod
	    def sigmoidprime(x):
			   return (__class__.sigmoid(x))*(1 - __class__.sigmoid(x))
			   
net = Net(cfg.Structure,"name")

#net.forwardPropagate();

conn = sqlite3.connect(cfg.DbConfig.DIR + cfg.DbConfig.NAME)

conn.close