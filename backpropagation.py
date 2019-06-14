# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
import numpy as np
import copy as cp

debug = 0


class Neuron:
    # basic type
    # set value
    # set sum
    
    value = 0
    sum = 0
    is_bias = 0
    weights = None
    deltaSum = None
    
    def __init__(self, is_bias=False):
        self.is_bias = is_bias
        return None
    
    def setValue(self, value):
        if self.is_bias:
            raise Exception("Bias value or sum cannot be set")
        self.value = value
        if debug:
            print("   I assign a value of " + str(value) + " to the neuron " + str(self))
        return self
    
    def getValue(self):
        if self.is_bias:
            return 1
        return self.value
    
    def setSum(self, sum):
        if self.is_bias:
            raise Exception("Bias value or sum cannot be set")
        self.sum = sum
        if debug:
            print("   I assign a sum of " + str(sum) + " to the neuron " + str(self))
    
    def getSum(self):
        if self.is_bias:
            return 1
        return self.sum
    
    def setWeights(self, weight):
        self.weights = np.array(weight)
        return self
    
    def getWeights(self):
        return self.weights

    def setDeltaSum(self, deltasum):
        self.deltaSum = deltasum
        return self

    def getDeltaSum(self):
        return self.deltaSum

class Layer:
    #
    # Get number of neurons
    # set neurons' values
    #
    neurons = {}
    layerSum = 0
    weights = {}
    bias = None
    
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
    
    def setBias(self, weights):
        bias = Neuron(True)
        bias.setWeights(weights)
        self.bias = bias
        return self
    
    def getBias(self):
        return self.bias
    
    def getBiasWeights(self):
        if self.bias != None:
            return np.array(self.bias.getWeights()).T
        
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
        self.weights = np.array(weights)
        for i in range(0, len(weights)):
            self.getNeuron(i).setWeights(weights[i])
        return self

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

    def __init__(self, name, inputs, results, learning_rate=1):
        self.setName(name)
        self.setInputs(inputs)
        self.setResults(results)
        self.learning_rate = learning_rate
        
    def setLayer(self, index, layer):
        self.layers[index] = layer
        return self

    def getLayer(self, index):
        return self.layers[index]
    
    def getLayers(self):
        return self.layers
    
    def setWeights(self, weights={}):
        if weights:
            self.weights = np.array(weights, dtype=object)
            for i in range(0, self.getNeurons()):
                self.getNeurons(i).setWeights(self.weights[i])
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
                    if weights:
                        for i in range(0, neuronsCount):
                            self.layers[index].getNeuron(i).setWeights(np.random.rand(1, neurons2Count))
                    else:
                        self.layers[index].setWeights(np.random.rand(neuronsCount, neurons2Count))
        return self
    
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
        if np.size(self.results) != np.size(self.getLayer(len(self.getLayers())-1).getValues()):
            # print(np.shape(self.results))
            # print(np.shape(self.getLayer(len(self.getLayers())-1).getNeurons()))
            #check if layer has bias, then recalculate
            # raise Exception('Results size must match last layer network!')
            pass
        for i in range(0, len(self.getLayers()) - 1):
            currentLayer = self.getLayer(i)
            nextLayer = self.getLayer(i+1)
            if debug:
                print("I set the " + str(i) + " layer: " + str(nextLayer) + " of network " + str(self))
            # produce neurons' sums and values
            for j in range(0, len(nextLayer.getNeurons())):
                sum = np.dot(nextLayer.getWeights()[j], currentLayer.getValues())
                if currentLayer.getBias() != None:
                    biasWeightsSum = currentLayer.getBiasWeights()[j] * 1
                    sum += biasWeightsSum
                nextLayer.getNeuron(j).setSum(sum)
                nextLayer.getNeuron(j).setValue(ActivationFn().sigmoid(sum))
        self.error = self.results - self.getLayer(i+1).getValues()
        if np.shape(self.error) != np.shape(self.results):
            raise Exception('Error must have the same shape as the results')
        print(self.getLayer(i + 1).getValues())

    def backPropagate(self):
        oldweights = cp.deepcopy(self.getWeights())
        for j in range(len(self.getWeights()), 0, -1):
            for ds in range(0, len(self.getLayer(j).getNeurons())):
                if j == len(self.getWeights()):
                    deltaSum = ActivationFn().sigmoidprime(self.getLayer(j).getNeuron(ds).getSum()) * self.error[ds]
                    self.getLayer(j).getNeuron(ds).setDeltaSum(deltaSum)
                else:
                    deltaSum = 0
                    for neur in range(0, len(self.getLayer(j+1).getNeurons())):
                        deltaSum += self.getLayer(j+1).getNeuron(neur).getDeltaSum() * oldweights[j+1][neur][ds]
                    self.getLayer(j).getNeuron(ds).setDeltaSum(deltaSum)
                deltaWeights = deltaSum * self.getLayer(j - 1).getValues()
                self.getLayer(j).weights[ds] = self.getLayer(j).weights[ds] + self.learning_rate * deltaWeights.T
                self.getLayer(j).getNeuron(ds).setWeights(self.layers[j].weights[ds] + self.learning_rate * deltaWeights.T)
                #update bias weights
                #it seems that possibly bias doesn't have to be updated
                # or it should be do along with the weights of the neurons
                if self.getLayer(j-1).getBias():
                    bias = self.getLayer(j-1).getBias()
                    bias.weights[ds] = bias.weights[ds] + (self.learning_rate * self.getLayer(j).getNeuron(ds).getDeltaSum() * self.getLayer(j).getNeuron(ds).getValue())

class ActivationFn:
    @staticmethod
    def sigmoid(x):
        x = np.array(x)
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoidprime(x):
        return (__class__.sigmoid(x)) * (1 - __class__.sigmoid(x))