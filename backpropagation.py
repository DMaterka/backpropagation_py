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
    weights = {}
    
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
    
    def setWeights(self, weights):
        # sh = shape(weights)
        self.weights = np.array(weights)
        size = np.shape(self.weights)
        if(size[1] != len(self.getNeurons())):
            Exception('Malformed weights')

    def getWeights(self):
        return self.weights

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

    def __init__(self, inputs, results, name):
        # self.setInputs(inputs)
        # self.results = results
        self.setName(name)
        self.setInputs(inputs)
        self.setResults(results)
        # TODO losowe wagi
        # self.weights[1] = np.array([[0.8, 0.2], [0.4, 0.9], [0.5, 0.3]])
        # self.weights[2] = np.array([[0.3, 0.5, 0.9]])
        
    def setLayer(self, index, layer):
        self.layers[index] = layer
        return self

    def getLayer(self, index):
        return self.layers[index]
    
    def getLayers(self):
        return self.layers
    
    # @TODO WIP
    def setWeights(self, weights={}):
        if(weights == {}):
            len(self.getLayers())
            test = {}
            for index, value in self.getLayers().items():
                if(index != len(self.getLayers()) - 1):
                    # @TODO too less Neurons
                    neuronsCount = len(value.getNeurons())
                    tt = self.getLayer(index + 1)
                    neurons2Count = len(self.getLayer(index+1).getNeurons())
                    test[index] = np.random.rand(neuronsCount, neurons2Count)
                    t =1
            return True
    
    def getWeights(self, layer=0) -> np.array:
        if(layer):
            return self.getLayer(layer).getWeights()
        else:
            layers = self.getLayers()
            weights = {}
            for layer in range(1, len(layers)):
                weights[layer] = layers[layer-1].getWeights()
            return weights
        
    def setInputs(self, inputs):
        self.inputs = inputs
        return self
        
    def getInputs(self):
        return self.inputs

    def setResults(self, results):
        self.results = results
        return self
        
    def getResults(self):
        return self.results
    
    def setDimensionsDef(self):
        inputsNumber = len(self.inputs)
        outputsNumber = len(self.results)
        hiddenNeurons = self.getLayer(1).getNeurons()
        self.dimensionsdef = [inputsNumber, len(hiddenNeurons), outputsNumber]
        
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
        for i in range(1, len(self.dimensionsdef) - 1):
            currentLayer = self.getLayer(i)
            previousLayer = self.getLayer(i-1)
            if debug:
                print("I set the " + str(i) + " layer: " + str(previousLayer) + " of network " + str(self))
            # produce neurons' sums
            values = np.dot(previousLayer.getWeights(), previousLayer.getValues())
            currentLayer.setNeurons(values)
        #set output value
        outputLayer = self.getLayer(i+1)
        if debug:
            print("I set the " + str(i + 1) + " layer: " + str(outputLayer) + " of network " + str(self))
        value = np.dot(self.getLayer(i).getWeights(), self.getLayer(i).getValues())
        outputLayer.setNeurons(value)
    
        # @TODO 0 replace with value of output neuron
        self.outputValue = outputLayer.getNeurons()[0].getValue()
        # print(self.outputValue)
        self.error = self.results - self.outputValue
        print(self.outputValue)

    def backPropagate(self):
        oldweights = cp.copy(self.getWeights())
        i = len(self.getWeights())
        deltaSum = ActivationFn().sigmoidprime(self.getLayer(i).getSums()[0]) * self.error
        deltaWeights = deltaSum / self.getLayer(i - 1).getValues()
        self.getLayer(i - 1).setWeights(self.getLayer(i - 1).getWeights() + deltaWeights)
        for j in range(len(self.getWeights()) - 1, 0, -1):
            deltaSum = deltaSum / oldweights[j + 1] * ActivationFn().sigmoidprime(self.getLayer(j).getSums())
            # allow legally divise the arrays with different shapes, use instead for over all inputs
            a = np.tile(deltaSum, (np.size(self.getLayer(j - 1).getValues()), 1)).T
            b = self.getLayer(j - 1).getValues().reshape(1, 2)
            # issue with 0.0 and 1.0 the weights doesn't move or the division by 0 occurs
            # that means the values are guessed well initially, so the network is confused
            # maybe sums should be distributed randomly?
            deltaWeigths = a / b
            self.getLayer(j - 1).setWeights(self.getLayer(j-1).getWeights() + deltaWeigths)

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

numit = 1000
hiddenNeuronsNumber = 3
# inputs = np.array([[0.01, 0.99], [0.01, 0.01], [0.99, 0.99], [0.99, 0.01]])
# results = np.array([[0.99], [0.01], [0.01], [0.99]])

inputs = np.array([[0.99, 0.99]])
results = np.array([[0.01]])

#train the network
net = Net(inputs, results, "name")
#set input layer
inputLayer = Layer()
inputLayer.setNeurons(inputs)
net.setLayer(0, inputLayer)
net.getLayer(0).setWeights([[0.8, 0.2], [0.4, 0.9], [0.5, 0.3]])
# net.getLayer(0).setWeights()
# set hidden layer
hiddenLayer = Layer()
hiddenLayer.setNeurons([0, 0, 0])
neurons = hiddenLayer.getNeurons()
net.setLayer(1, hiddenLayer)
net.getLayer(1).setWeights([[0.3, 0.5, 0.9]])
#set output layer
net.setLayer(2, Layer())
net.setWeights()
for i in range(1, 10):
    for pair in range(0, len(inputs)):
        for i in range(1, numit):
            net.setInputs(inputs[pair])
            net.setResults(results[pair])
            net.setDimensionsDef()
            net.forwardPropagate()
            net.backPropagate()
            
net.setInputs(inputs[0])
net.forwardPropagate()
# net.setInputs(inputs[1])
# net.forwardPropagate()
#validate the network

#store the network
conn = sqlite3.connect(cfg.DbConfig.DIR + cfg.DbConfig.NAME)

conn.close
