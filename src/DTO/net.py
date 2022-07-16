import numpy as np
from .layer import Layer
import os


class Net:
    """ contains layers
    make a forward and backpropagation
    return layer at any moment
    :var name is well formatted csv file, it contains data in columns, where the last column is expected result
    """

    def __init__(self, name: str, learning_rate=0.5):
        self.learning_curve_data = []
        self.learning_progress = []
        self.layers = []
        self.setName(name)
        self.learning_rate = learning_rate
        self.debug = os.getenv('DEBUG', False) == True

    def setLayer(self, index, layer: Layer):
        self.getLayers().insert(index, layer)
        # if len(layer.getWeights()) > 0:
        # if np.shape(layer.getWeights()) != (len(layer.getNeurons()), len(self.getLayer(index-1).getNeurons())):
        #     raise Exception("Bad weights format")
        return self

    def getLayer(self, index):
        return self.layers[index]

    def getLayers(self):
        return self.layers

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