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
        self.deltaSums = []
    
    def setNeurons(self, sums):
        sums = np.array(sums)
        self.neurons = [Neuron() for i in range(0, len(sums))]
        for i in range(len(sums)):
            if debug:
                print("  I insert a " + str(i) + " neuron: " + str(self.neurons[i]) + " of layer " + str(self))
            self.neurons[i].setSum(sums[i])
            self.neurons[i].setValue(ActivationFn().sigmoid(sums[i]))
           
        return self
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
        for i in range(len(self.neurons)):
            values.append(self.neurons[i].getValue())
        values = np.array(values)
        return values
    
    def setValues(self, values):
        """ Set values of the layer's neurons"""
        for i in range(len(self.neurons)):
            self.neurons[i].setValue(ActivationFn().sigmoid(values[i]))
        return self
    
    def setSums(self, sums):
        """ Set sums of the layer's neurons"""
        for i in range(len(self.neurons)):
            self.neurons[i].setSum(sums[i])
        return self
    
    def getSums(self):
        """ Get neurons sums in the layer"""
        tmparr = np.array([])
        for i in range(len(self.neurons)):
            tmparr = np.append(tmparr, self.neurons[i].getSum())
        return tmparr
    
    def setWeights(self, weights: list):
        self.weights = np.array(weights)
        for i in range(len(weights)):
            self.getNeuron(i).setWeights(weights[i])
        return self

    def getWeights(self):
        return self.weights
    
    def setDeltaSums(self, deltasums):
        self.deltaSums = deltasums
        return self
    
    def getDeltaSums(self):
        """ Get values of the layer's neurons"""
        deltasums = []
        for i in range(len(self.getNeurons())):
            deltasums.append(self.getNeuron(i).getDeltaSum())
        return np.array(deltasums)

class Net:
    """ contains layers
    make a forward and backpropagation
    return layer at any moment
    :var name is well formatted csv file, it contains data in columns, where the last column is expected result
    """

    def __init__(self, name: str, learning_rate=0.5):
        df = pd.read_csv('data/' + name)
        self.layers = []
        self.setName(name)
        self.learning_rate = learning_rate
        # set input layer
        inputLayer = Layer()
        inputLayer.setNeurons(df.iloc[:, :].values.T)
        self.setLayer(0, inputLayer)
        
    def setLayer(self, index, layer):
        self.getLayers().insert(index, layer)
        # if len(layer.getWeights()) > 0:
            # if np.shape(layer.getWeights()) != (len(layer.getNeurons()), len(self.getLayer(index-1).getNeurons())):
            #     raise Exception("Bad weights format")
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
            for layer in range(len(layers)):
                weights[layer] = layers[layer].getWeights()
            return weights

    def setInputs(self, inputs):
        self.inputs = inputs
        return self
        
    def getInputs(self):
        return self.getLayer(0).getValues()
    
    def setExpectedResults(self, expected_results):
        self.expected_results = np.array(expected_results)
        return self
        
    def getExpectedResults(self):
        return self.expected_results
    
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
        for i in range(0, len(self.getLayers()) - 1):
            currentLayer = self.getLayer(i)
            nextLayer = self.getLayer(i+1)
            
            """ produce neurons' sums and values """
            for j in range(len(nextLayer.getNeurons())):
                sum = 0
                weights = nextLayer.getNeuron(j).getWeights()
                
                if i == 0:
                    values = currentLayer.getSums()
                else:
                    values = currentLayer.getValues()
                    
                sum += np.dot(weights, values)
                
                if currentLayer.getBias() is not None:
                    biasWeightsSum = currentLayer.getBiasWeights()[j] * 1
                    sum += biasWeightsSum
                nextLayer.getNeuron(j).setSum(sum)
                nextLayer.getNeuron(j).setValue(ActivationFn().sigmoid(sum))

    def calculateTotalError(self, inputs):
        total_error = 0
        for index in range(len(inputs)):
            init, expected = inputs[index]
            self.getLayer(0).setNeurons(init)
            self.setExpectedResults(expected)
            self.forwardPropagate()
            total_error += np.sum((0.5 * ((self.getExpectedResults().T - self.get_results())) ** 2))
        return total_error
        
    def backPropagate(self):
        oldSelf = cp.deepcopy(self)
        self.forwardPropagate()
        for j in range(len(self.getLayers()) - 1, 0, -1):
            for ds in range(len(self.getLayer(j).getWeights())):
                if j == len(self.getLayers()) - 1:
                    partial_error = self.getLayer(j).getValues() - self.getExpectedResults().T
                    deltaSum = ActivationFn().sigmoidprime(self.getLayer(j).getNeuron(ds).getSum()) * partial_error[0][ds] * self.getLayer(j-1).getNeuron(ds).getValue()
                else:
                    upper_layer_delta_sums = self.getLayer(j+1).getDeltaSums()
                    current_neuron_values = self.getLayer(j).getValues()
                    t = upper_layer_delta_sums / current_neuron_values
                    partial_sum = 0
                    for up_neur in range(len(self.getLayer(j+1).getNeurons())):
                        weight = self.getLayer(j + 1).getNeuron(up_neur).getWeights()[ds]
                        partial_sum += t[up_neur] * weight
                    
                    d_val = ActivationFn().sigmoidprime(self.getLayer(j).getNeuron(ds).getSum())
                    deltaSum = partial_sum * d_val * self.getLayer(j-1).getNeuron(ds).getValue()

                self.getLayer(j).getNeuron(ds).setDeltaSum(deltaSum)

        for j in range(len(self.getLayers()) - 1, 0, -1):
            for ds in range(len(self.getLayer(j).getWeights())):
                new_weight = oldSelf.getLayer(j).getNeuron(ds).getWeights() - (self.learning_rate * self.getLayer(j).getNeuron(ds).getDeltaSum())
                self.getLayer(j).getNeuron(ds).setWeights(new_weight)
        
        # if self.getLayer(j).getBias():
            # bias = self.getLayer(j-1).getBias()
            # biasValue = bias.weights[ds] + (
            #             self.learning_rate * self.getLayer(j).getNeuron(ds).getDeltaSum() *
            #             self.getLayer(j).getNeuron(ds).getValue()
            # )
            # np.append(bias.weights, biasValue)
                    
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

    def get_results(self):
        return self.getLayer(len(self.getLayers()) - 1).getValues()
    
        
class ActivationFn:
    @staticmethod
    def sigmoid(x):
        x = np.array(x)
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoidprime(x):
        return (__class__.sigmoid(x)) * (1 - __class__.sigmoid(x))

    @staticmethod
    def tanh(z):
        return np.tanh(z)

    @staticmethod
    def relu(z):
        if z > 0:
            return input
        return 0

    @staticmethod
    def relu_prime(z):
        if z > 0:
            return 1
        return 0
