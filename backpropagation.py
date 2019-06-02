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
    bias = 0
    
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
    
    def setBias(self, bias):
        self.bias = bias
        return self
    
    def getBias(self):
        return self.bias


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
        t = np.resize(sums, (np.size(sums), 1))
        self.neurons = [Neuron() for i in range(len(t))]
        for i in range(len(t)):
            if debug:
                print("  I insert a " + str(i) + " neuron: " + str(self.neurons[i]) + " of layer " + str(self))
            self.neurons[i].setSum(t[i])
            if immutable != 1:
                self.neurons[i].setValue(ActivationFn().sigmoid(t[i]))
            else:
                self.neurons[i].setValue(t[i])
    
    def getNeurons(self):
        return self.neurons
    
    def getNeuron(self, index):
        return self.neurons[index]
    
    def getValues(self):
        tmparr = np.zeros(np.shape(self.neurons))
        for i in range(0, len(self.neurons)):
            tmparr[i] = self.neurons[i].getValue()
        return np.expand_dims(tmparr, 1)
    
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
        self.setName(name)
        self.setInputs(inputs)
        self.setResults(results)
        
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
                if index > 0:
                    neuronsCount = len(value.getNeurons())
                    neurons2Count = len(self.getLayer(index-1).getNeurons())
                    if neurons2Count == 0:
                        neurons2Count = 1
                    if neuronsCount == 0:
                        neuronsCount = 1
                    self.layers[index].setWeights(np.random.rand(neuronsCount, neurons2Count))
                    
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
        # validate network
        if len(self.results) != len(self.getLayer(len(self.getLayers())-1).getNeurons()):
            print(np.shape(self.results))
            print(np.shape(self.getLayer(len(self.getLayers())-1).getNeurons()))
            raise Exception('Results size must match last layer network!')
        
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
        self.error = self.results - self.getLayer(i+1).getValues()
        if np.shape(self.error) != np.shape(self.results):
            raise Exception('Error must have the shape of results')
        print(self.getLayer(i + 1).getValues())

    def backPropagate(self):
        oldweights = cp.copy(self.getWeights())
        i = len(self.getWeights())
        for ds in range(0, len(self.getLayer(i).getNeurons())):
            deltaSum = ActivationFn().sigmoidprime(self.getLayer(i).getNeuron(ds).getSum()) * self.error[ds]
            t1 = self.getLayer(i - 1).getValues()
            deltaWeights = deltaSum / self.getLayer(i - 1).getValues()
            t2 = self.layers[i].weights
            self.layers[i].weights[ds] = self.layers[i].weights[ds] + deltaWeights.T
        for j in range(len(self.getWeights()) - 1, 0, -1):
            for ds in range(0, len(self.getLayer(j).getNeurons())-1):
                deltaSum = deltaSum / oldweights[j+1][0][ds] * ActivationFn().sigmoidprime(self.getLayer(j).getNeuron(ds).getSum())
                deltaWeigths1 = deltaSum / self.getLayer(j - 1).getValues()
                self.layers[j].weights[ds] = self.layers[j].weights[ds] + deltaWeigths1.T


class ActivationFn:
    @staticmethod
    def sigmoid(x):
        x = np.array(x)
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoidprime(x):
        return (__class__.sigmoid(x)) * (1 - __class__.sigmoid(x))

#store the network
conn = sqlite3.connect(cfg.DbConfig.DIR + cfg.DbConfig.NAME)

conn.close
