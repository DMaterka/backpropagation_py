# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
import numpy as np
import copy as cp
import matplotlib.pyplot as plt
import numpy.matlib
import pandas as pd

debug = 0


class Neuron:
    """
    basic type
    set value
    set sum
    """
    
    def __init__(self, is_bias=False):
        self.is_bias = is_bias
        self.value = 0
        self.sum = 0
        self.is_bias = 0
        self.weights = []
        self.deltaSum = None
        self.position = None
    
    def setValue(self, value):
        if self.is_bias:
            raise Exception("Bias value or sum cannot be set")
        self.value = np.array(value)
        if debug:
            print("   I assign a value of " + str(self.value) + " to the neuron " + str(self))
        return self
    
    def getValue(self):
        if self.is_bias:
            return 1
        return self.value
    
    def setSum(self, sum):
        if self.is_bias:
            raise Exception("Bias value or sum cannot be set")
        self.sum = np.array(sum)
        if debug:
            print("   I assign a sum of " + str(self.sum) + " to the neuron " + str(self))
    
    def getSum(self):
        if self.is_bias:
            return 1
        return self.sum
    
    def setWeights(self, weight, index=None):
        if index is not None:
            self.weights.insert(index, weight)
        else:
            self.weights = np.array(weight)
        return self
    
    def getWeights(self):
        return np.array(self.weights)

    def setDeltaSum(self, deltasum):
        self.deltaSum = deltasum
        return self

    def getDeltaSum(self):
        return self.deltaSum
    
    def setPosition(self, position: list):
        self.position = position
        return self


class Layer:
    """
    Get number of neurons
    set neurons' values
    """

    def __init__(self):
        self.neurons = []
        self.weights = []
        self.bias = None
    
    def setNeurons(self, sums, immutable=0):
        sums = np.array(sums)
        self.neurons = [Neuron() for i in range(0, len(sums))]
        for i in range(0, len(sums)):
            if debug:
                print("  I insert a " + str(i) + " neuron: " + str(self.neurons[i]) + " of layer " + str(self))
            self.neurons[i].setSum(sums[i])
            if immutable != 1:
                self.neurons[i].setValue(ActivationFn().sigmoid(sums[i]))
            else:
                self.neurons[i].setValue(sums[i])
    
    def getNeurons(self):
        return self.neurons
    
    def setNeuron(self, index, neuron):
        self.neurons.insert(index, neuron)
        return self
    
    def getNeuron(self, index: int):
        return self.neurons[index]
    
    def setBias(self, weights):
        bias = Neuron(True)
        bias.setWeights(weights)
        self.bias = bias
        return self
    
    def getBias(self):
        return self.bias
    
    def getBiasWeights(self):
        if self.bias is not None:
            return np.array(self.bias.getWeights()).T
        
    def getValues(self):
        """ Get values of the layer's neurons"""
        values = []
        for i in range(0, len(self.neurons)):
            values.append(self.neurons[i].getValue())
        values = np.array(values)
        if np.ndim(values) == 1:
            values = np.expand_dims(values, 1)
        return values
    
    def getSums(self):
        """ Get neurons sums in the layer"""
        tmparr = np.array([])
        for i in range(len(self.neurons)):
            tmparr = np.append(tmparr, self.neurons[i].getSum())
        return tmparr
    
    def setWeights(self, weights: list):
        self.weights = np.array(weights)
        for i in range(0, len(weights)):
            self.getNeuron(i).setWeights(weights[i])
        return self

    def getWeights(self):
        return self.weights


class Net:
    """ contains layers
    make a forward and backpropagation
    return layer at any moment
    :var name is well formatted csv file, it contains data in columns, where the last column is expected result
    """

    def __init__(self, name: str, learning_rate=1):
        df = pd.read_csv('data/' + name)
        self.layers = []
        self.error = 0
        self.setName(name)
        self.setResults(df.iloc[:, -1:].values.T)
        self.learning_rate = learning_rate
        self.dims = 0
        self.name = ''
        self.weights = {}
        self.deltaOutputSum = 0
        # set input layer
        inputLayer = Layer()
        inputLayer.setNeurons(df.iloc[:, :-1].values.T, 1)
        self.setLayer(0, inputLayer)
        
    def setLayer(self, index, layer):
        self.getLayers().insert(index, layer)
        if len(layer.getWeights()) > 0:
            if np.shape(layer.getWeights()) != (len(layer.getNeurons()), len(self.getLayer(index-1).getNeurons())):
                raise Exception("Bad weights format")
        return self

    def getLayer(self, index):
        return self.layers[index]
    
    def getLayers(self):
        return self.layers
    
    # deprecated
    def setWeights(self, weights={}):
        if weights:
            self.weights = np.array(weights, dtype=object)
            for i in range(0, self.getNeurons()):
                self.getNeurons(i).setWeights(self.weights[i])
        #         to delete ?
        if weights == {}:
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
                        self.layers[index].setWeights(np.random.rand(len(self.inputs), neuronsCount, neurons2Count))
        return self
    
    def getWeights(self, layer=0) -> np.array:
        if layer:
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
        return self.getLayer(0).getValues()

    def setResults(self, results):
        self.results = np.array(results)
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
        """ calculate network values from weights and activation function"""
        if np.size(self.results) != np.size(self.getLayer(0).getNeuron(0).getValue()):
            raise Exception('Results size must match input neuron size!')
        for i in range(0, len(self.getLayers()) - 1):
            currentLayer = self.getLayer(i)
            nextLayer = self.getLayer(i+1)
            if debug:
                print("I set the " + str(i) + " layer: " + str(nextLayer) + " of network " + str(self))
            """ produce neurons' sums and values """
            for j in range(0, len(nextLayer.getNeurons())):
                weights = nextLayer.getNeuron(j).getWeights()
                values = currentLayer.getValues()
                
                sum = np.dot(weights.T, values)
                if np.ndim(sum) > 1:
                    sum = np.mean(sum, 1)
                
                if currentLayer.getBias() is not None:
                    biasWeightsSum = currentLayer.getBiasWeights()[j] * 1
                    sum += biasWeightsSum
                nextLayer.getNeuron(j).setSum(sum)
                nextLayer.getNeuron(j).setValue(ActivationFn().sigmoid(sum))
        self.error = self.results - self.getLayer(i+1).getValues()
        #error must have the dimensions 'flat'
        if np.shape(self.error) != np.shape(self.results):
            pass
            # raise Exception('Error must have the same shape as the results')

    def backPropagate(self):
        oldSelf = cp.deepcopy(self)
        for j in range(len(self.getWeights()), 0, -1):
            for ds in range(0, len(self.getLayer(j).getNeurons())):
                if j == len(self.getWeights()):
                    deltaSum = ActivationFn().sigmoidprime(self.getLayer(j).getNeuron(ds).getSum()) * self.error[ds]
                    self.getLayer(j).getNeuron(ds).setDeltaSum(deltaSum)
                else:
                    deltaSum = 0
                    for neur in range(0, len(self.getLayer(j+1).getNeurons())):
                        deltaSum += self.getLayer(j+1).getNeuron(neur).getDeltaSum() * \
                                    oldSelf.getLayer(j+1).getNeuron(neur).getWeights()[ds]
                    self.getLayer(j).getNeuron(ds).setDeltaSum(deltaSum)
                deltaWeights = self.getLayer(j).getNeuron(ds).getDeltaSum() * self.getLayer(j - 1).getValues()
                weights = self.getLayer(j).getNeuron(ds).getWeights()
                rows_number = np.shape(self.getInputs())[1]
                
                if np.shape(weights) != np.shape(deltaWeights):
                    self.getLayer(j).getNeuron(ds).setWeights(
                        np.matlib.repmat(self.getLayer(j).getNeuron(ds).getWeights(), rows_number, 1).T
                    )
                    
                newWeights = self.getLayer(j).getNeuron(ds).getWeights() + (self.learning_rate * deltaWeights)
                
                self.getLayer(j).getNeuron(ds).setWeights(newWeights)
                #update bias weights
                #it seems that possibly bias doesn't have to be updated
                # or it should be do along with the weights of the neurons
                if self.getLayer(j-1).getBias():
                    bias = self.getLayer(j-1).getBias()
                    biasValue = bias.weights[ds] + (
                                self.learning_rate * self.getLayer(j).getNeuron(ds).getDeltaSum() * self.getLayer(
                            j).getNeuron(ds).getValue())
                    np.append(bias.weights, biasValue)
                    
    def print_network(self):
        fig, axs = plt.subplots()
        axs.set_xlim((0, 100))
        axs.set_ylim((0, 100))
        posx = 10
        radius = 10
        for layer_index in range(len(self.getLayers())):
            interval = 100 / (len(
                self.getLayer(layer_index).getNeurons()) + 1)
            posy = interval
            for neuron_index in range(len(self.getLayer(layer_index).getNeurons())):
                axs.add_artist(plt.Circle((posx, posy), radius))
                text_to_show = 'sum:' + '{:.2f}'.format(self.getLayer(layer_index).getNeuron(neuron_index).getSum()[0])
                text_to_show += "\n" + 'value:' + "{:.2f}".format(
                    self.getLayer(layer_index).getNeuron(neuron_index).getValue()[0]
                )
                plt.text(posx, posy, text_to_show, fontsize=12)
                if layer_index > 0:
                    weights = ''
                    for weight_index in range(0, len(self.getLayer(layer_index).getNeuron(neuron_index).getWeights())):
                        weight_value = self.getLayer(layer_index).getNeuron(neuron_index).getWeights()[weight_index]
                        if np.ndim(weight_value) == 0:
                            weight_value = np.expand_dims(
                                self.getLayer(layer_index).getNeuron(neuron_index).getWeights()[weight_index], 1
                            )
                        weights += '{:.2f}'.format(weight_value[0]) + "\n"
                    plt.text(posx - radius, posy-0.5*interval, weights, fontsize=12)
                posy += interval
                self.getLayer(layer_index).getNeuron(neuron_index).setPosition([posx, posy])
            posx += radius*4
            
        fig.tight_layout()
    
        plt.show()
        
        
class ActivationFn:
    @staticmethod
    def sigmoid(x):
        x = np.array(x)
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoidprime(x):
        return (__class__.sigmoid(x)) * (1 - __class__.sigmoid(x))