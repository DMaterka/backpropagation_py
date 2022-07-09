debug=0
import numpy as np


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
            print(" I assign a sum of " + str(self.sum) + " to the neuron " + str(self))

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
