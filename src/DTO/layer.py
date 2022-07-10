import numpy as np
from .neuron import Neuron
from ..activation import ActivationFn

debug=0


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

    def setNeurons(self, sums, activate=True):
        sums = np.array(sums)

        self.neurons = [Neuron() for i in range(0, len(sums))]
        for i in range(len(sums)):
            if debug:
                print("  I insert a " + str(i) + " neuron: " + str(self.neurons[i]) + " of layer " + str(self))
            self.neurons[i].setSum(sums[i])
            if activate == True:
                self.neurons[i].setValue(ActivationFn().sigmoid(sums[i]))
            else:
                self.neurons[i].setValue(sums[i])

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

        if len(self.neurons) != len(weights):
            raise Exception(
                "Bad weights size. Expected weights size is the same as neurons size = " + str(len(self.neurons))
            )

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