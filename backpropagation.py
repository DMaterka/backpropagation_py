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
    # basic type
    # set value
    # set sum
    
    value = 0
    sum = 0
    
    def __init__(self):
        pass
    
    def setValue(self, value):
        self.value = value
        if debug:
            print("   I assign a value of " + str(value) + " to the neuron " + str(self))
    
    def getValue(self):
        return self.value
    
    def setSum(self, sum):
        self.sum = sum
        if debug:
            print("   I assign a sum of " + str(sum) + " to the neuron " + str(self))
    
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
    
    def setNeurons(self, sums, immutable=0):
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
            tmparr = np.append(tmparr, self.neurons[i].getValue())
        return tmparr
    
    def getSums(self):
        tmparr = np.array([])
        for i in (range(len(self.neurons))):
            tmparr = np.append(tmparr, self.neurons[i].getSum())
        return tmparr


class Net:
    # contains layers
    # make a forward and backpropagation
    # return layer at any moment
    dims = 0
    name = ''
    dimensionsdef = []
    weights = {}
    layers = {}
    deltaOutputSum = 0
    error = 0
    outputValue = 0

    def __init__(self, hiddenNeurons, name):
        # self.setInputs(inputs)
        # self.results = results
        # TODO losowe wagi
        self.weights[1] = np.array([[0.8, 0.2], [0.4, 0.9], [0.5, 0.3]])
        self.weights[2] = np.array([[0.3, 0.5, 0.9]])
        self.setName(name)
        self.hiddenNeurons = hiddenNeuronsNumber
        
    def setInputs(self, inputs):
        self.inputs = inputs
        t = self.inputs.shape
        
        inputsPairsNumber = self.inputs.shape[0]
        # assuming 1 layer of hidden neurons
        
        # without output layer
        layersNumber = 2
        
        
    def getInputs(self):
        return self.inputs

    def setResults(self, results):
        self.results = results
        
    def getResults(self):
        return self.results
    
    def setDimensionsDef(self):
        inputsNumber = len(self.inputs)
        outputsNumber = len(self.results)
        self.dimensionsdef = [inputsNumber, self.hiddenNeurons, outputsNumber]
        
    def getDimensionsDef(self):
        return self.dimensionsdef
    
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
        # @TODOset only first time
        self.layers[0].setNeurons(self.inputs, 1)
    
        for i in range(1, len(self.dimensionsdef) - 1):
            self.layers[i] = Layer()
            if debug:
                print("I set the " + str(i) + " layer: " + str(self.layers[i - 1]) + " of network " + str(self))
            # produce neurons' sums
            values = np.dot(self.weights[i], self.layers[i - 1].getValues())
            self.layers[i].setNeurons(values)
    
        self.layers[i + 1] = Layer()
        if debug:
            print("I set the " + str(i + 1) + " layer: " + str(self.layers[i]) + " of network " + str(self))
        value = np.dot(self.weights[len(self.weights)], self.layers[i].getValues())
        self.layers[i + 1].setNeurons(value)
    
        # @TODO 0 replace with value of output neuron
        self.outputValue = self.layers[i + 1].getNeurons()[0].getValue()
        # print(self.outputValue)
        self.error = self.results - self.outputValue
        print(self.error)

    def backPropagate(self):
        oldweights = cp.copy(self.weights)
        i = len(self.weights)
        deltaSum = ActivationFn().sigmoidprime(self.layers[i].getSums()[0]) * self.error
        deltaWeights = deltaSum / self.layers[i - 1].getValues()
        self.weights[i] = self.weights[i] + deltaWeights
    
        for j in range(len(self.weights) - 1, 0, -1):
            deltaSum = deltaSum / oldweights[j + 1] * ActivationFn().sigmoidprime(self.layers[(j)].getSums())
            # allow legally divise the arrays with different shapes, use instead for over all inputs
            a = np.tile(deltaSum, (np.size(self.layers[(j - 1)].getValues()), 1)).T
            b = self.layers[(j - 1)].getValues().reshape(1, 2)
            # issue with 0.0 and 1.0 the weights doesn't move or the division by 0 occurs
            # that means the values are guessed well initially, so the network is confused
            # maybe sums should be distributed randomly?
            deltaWeigths = a / b
            self.weights[j] = self.weights[j] + deltaWeigths
            oldweights = cp.copy(self.weights)
        # print(self.weights)

# def getOutputValue(self):
# 	   print('TODO')


class ActivationFn():
    @staticmethod
    def sigmoid(x):
        x = np.array(x)
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoidprime(x):
        return (__class__.sigmoid(x)) * (1 - __class__.sigmoid(x))


# test = np.array([[-0.03325106, -0.13300425],[-0.06650212, -0.0295565 ],[-0.0532017, -0.0886695 ]])
# print(test)
# print(test.T)
# print(test / np.array([1, 1]) )
# dane testowe oraz wynik jako jeden wiersz tablicy
# reprezentowane przez XOR

numit = 10000
hiddenNeuronsNumber = 3
inputs = np.array([[0.01, 0.99], [0.01, 0.01], [0.99, 0.99], [0.99, 0.01]])
results = np.array([[1], [0.01], [0.01], [1]])
#train the network
net = Net(hiddenNeuronsNumber, "name")
for i in range(1, numit):
    for pair in range(0, len(inputs)):
        net.setInputs(inputs[pair])
        net.setResults(results[pair])
        net.setDimensionsDef()
        net.forwardPropagate()
        net.backPropagate()

#validate the network

#store the network
conn = sqlite3.connect(cfg.DbConfig.DIR + cfg.DbConfig.NAME)

conn.close
