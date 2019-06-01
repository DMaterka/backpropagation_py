# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
import config as cfg
import numpy as np
import sqlite3
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
    
    def getNeuron(self, index):
        return self.neurons[index]
    
    def getValues(self):
        tmparr = np.zeros(np.shape(self.neurons))
        for i in (range(len(self.neurons))):
            tmparr[i] = self.neurons[i].getValue()
        return np.expand_dims(tmparr,1)
    
    def getSums(self):
        tmparr = np.array([])
        for i in (range(len(self.neurons))):
            tmparr = np.append(tmparr, self.neurons[i].getSum())
        return tmparr
    
    def setWeights(self, weights):
        # sh = shape(weights)
        self.weights = np.array(weights)
        # size = np.shape(self.weights)
        # if(size[1] != len(self.getNeurons())):
        #     Exception('Malformed weights')

    def getWeights(self):
        return self.weights

class Net:
    # contains layers
    # make a forward and backpropagation
    # return layer at any moment
    dims = 0
    name = ''
    weights = {}
    layers = {}
    deltaOutputSum = 0
    error = 0

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
        if (weights):
            self.weights = weights
            return self
        if(weights == {}):
            len(self.getLayers())
            for index, value in self.getLayers().items():
                if index < len(self.getLayers()) - 1:
                    neuronsCount = len(value.getNeurons())
                    neurons2Count = len(self.getLayer(index+1).getNeurons())
                    if neurons2Count == 0:
                        neurons2Count = 1
                    if neuronsCount == 0:
                        neuronsCount = 1
                    self.layers[index].setWeights(np.random.rand(neurons2Count, neuronsCount))
                    
    def getWeights(self, layer=0) -> np.array:
        if(layer):
            return self.getLayer(layer).getWeights()
        else:
            layers = self.getLayers()
            weights = {}
            for layer in range(1, len(layers)):
                weights[layer] = layers[layer].getWeights()
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
    
    def setDimensionNumber(self, dims):
        self.dims = dims

    def getDimensionNumber(self):
        return self.dims

    def setName(self, name):
        self.name = name

    def getName(self):
        return self.name

    def forwardPropagate(self):
        for i in range(0, len(self.getLayers()) - 1):
            currentLayer = self.getLayer(i)
            nextLayer = self.getLayer(i+1)
            if debug:
                print("I set the " + str(i) + " layer: " + str(nextLayer) + " of network " + str(self))
            # produce neurons' sums and values
            for j in range(0, len(nextLayer.getNeurons())):
                sum = np.dot(nextLayer.getWeights()[j], currentLayer.getValues())
                nextLayer.getNeuron(j).setSum(sum)
                nextLayer.getNeuron(j).setValue(ActivationFn().sigmoid(sum))
        self.error = self.results - self.getLayer(i+1).getNeurons()[0].getValue()
        print(self.getLayer(i + 1).getNeurons()[0].getValue())

    def backPropagate(self):
        oldweights = cp.copy(self.getWeights())
        i = len(self.getWeights())
        deltaSum = ActivationFn().sigmoidprime(self.getLayer(i).getSums()[0]) * self.error
        deltaWeights = deltaSum / self.getLayer(i - 1).getValues()
        self.getLayer(i).setWeights(self.getLayer(i).getWeights() + deltaWeights.T)
        for j in range(len(self.getWeights()) - 1, 0, -1):
            for ds in range(0, len(self.getLayer(j).getNeurons())-1):
                deltaSum = deltaSum / oldweights[j+1][0][ds] * ActivationFn().sigmoidprime(self.getLayer(j).getNeuron(ds).getSum())
                deltaWeigths1 = deltaSum / self.getLayer(j - 1).getValues()[ds]
                self.layers[j].weights[ds] = self.layers[j].weights[ds] + deltaWeigths1

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

numit = 2
inputs = np.array([[0.01, 0.99], [0.01, 0.01], [0.99, 0.99], [0.99, 0.01]])
results = np.array([[0.99], [0.01], [0.01], [0.99]])

# inputs = np.array([[0.99, 0.99]])
# results = np.array([[0.01]])

#train the network
net = Net(inputs, results, "name")
#set input layer
inputLayer = Layer()
inputLayer.setNeurons(inputs[2], 1)
net.setLayer(0, inputLayer)
# set hidden layer
hiddenLayer = Layer()
hiddenLayer.setNeurons([0, 0, 0])
net.setLayer(1, hiddenLayer)
net.getLayer(1).setWeights([[0.8, 0.2], [0.4, 0.9], [0.5, 0.3]])

# set second hidden layer
hiddenLayer1 = Layer()
hiddenLayer1.setNeurons([0, 0, 0, 0])
net.setLayer(2, hiddenLayer1)
net.getLayer(2).setWeights([[.3, .5, .9], [.5, .3, .6], [.8, .1, .6], [.5, .5, .4]])

#
# hiddenLayer2 = Layer()
# hiddenLayer2.setNeurons([0, 0, 0, 0, 0])
# net.setLayer(3, hiddenLayer2)
# net.getLayer(2).setWeights([[0.6, 0.8, 0.3, 0.7, 0.2]])

#set output layer
outputLayer = Layer()
outputLayer.setNeurons([0])
net.setLayer(3, outputLayer)
# net.getLayer(2).setWeights([[.3, .5, .9]])
net.getLayer(3).setWeights([[.3, .5, .9, .7]])
# net.setWeights()
t = len(inputs)
for pair in range(2, 3):
    for i in range(1, numit):
        # net.setInputs(inputs[pair])
        net.getLayer(0).setNeurons(inputs[pair], 1)
        net.setResults(results[pair])
        net.forwardPropagate()
        net.backPropagate()
            
print('results')
net.setInputs(inputs[0])
net.forwardPropagate()

net.setInputs(inputs[1])
net.forwardPropagate()

net.setInputs(inputs[2])
net.forwardPropagate()

net.setInputs(inputs[3])
net.forwardPropagate()
#validate the network

#store the network
conn = sqlite3.connect(cfg.DbConfig.DIR + cfg.DbConfig.NAME)

conn.close
