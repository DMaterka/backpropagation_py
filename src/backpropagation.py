import numpy as np

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
        self.learning_curve_data = None
        self.learning_progress = []
        self.layers = []
        self.setName(name)
        self.learning_rate = learning_rate

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
                if nextLayer.getNeuron(j).is_bias:
                    continue
                weights = nextLayer.getNeuron(j).getWeights()
                
                if i == 0:
                    values = currentLayer.getSums()
                else:
                    values = currentLayer.getValues()
                sum = np.dot(weights, values)
                nextLayer.getNeuron(j).setSum(sum)
                nextLayer.getNeuron(j).setValue(ActivationFn().sigmoid(sum))

    def calculateTotalError(self):
        total_error = 0
        for index in range(len(self.getExpectedResults())):
            total_error += np.sum((0.5 * (self.getExpectedResults()[index] - self.get_results()[index]) ** 2))
        self.learning_curve_data.append(total_error)
        return total_error
        
    def backPropagate(self):
        self.forwardPropagate()
        total_error = self.calculateTotalError()
        print(total_error)
        for j in range(len(self.getLayers()) - 1, 0, -1):
            for weight_index in range(len(self.getLayer(j).getWeights())):
                if j == len(self.getLayers()) - 1:
                    # this represents a value of partial derivative of results error dExp_results/dValues
                    partial_error = self.getLayer(j).getValues() - self.getExpectedResults().T
                    # deltaSum is partial derivative of total error with respect to given weight which consists of:
                    # partial derivative of next's neuron value with respect to the sum
                    # times partial error
                    # times next layer partial derivative of sum with respect to a weight
                    deltaSum = ActivationFn().sigmoidprime(self.getLayer(j).getNeuron(weight_index).getSum()) \
                               * partial_error[weight_index] \
                               * self.getLayer(j-1).getValues()
                else:
                    partial_sum = 0
                    for up_neur in range(len(self.getLayer(j+1).getNeurons())):
                        weight = self.getLayer(j + 1).getNeuron(up_neur).getWeights()[weight_index]
                        err_times_upper_delta = (
                                self.getLayer(j + 1).getNeuron(up_neur).getDeltaSum()[up_neur] /
                                self.getLayer(j).getNeuron(up_neur).getValue()
                        )
                        partial_sum += err_times_upper_delta * weight
                    
                    d_val = ActivationFn().sigmoidprime(self.getLayer(j).getNeuron(weight_index).getSum())
                    deltaSum = partial_sum * d_val * self.getLayer(j-1).getSums()

                self.getLayer(j).getNeuron(weight_index).setDeltaSum(deltaSum)

        self.update_weights()
        
        # if self.getLayer(j).getBias():
            # bias = self.getLayer(j-1).getBias()
            # biasValue = bias.weights[ds] + (
            #             self.learning_rate * self.getLayer(j).getNeuron(ds).getDeltaSum() *
            #             self.getLayer(j).getNeuron(ds).getValue()
            # )
            # np.append(bias.weights, biasValue)

    def update_weights(self):
        for j in range(len(self.getLayers()) - 1, 0, -1):
            for ds in range(len(self.getLayer(j).getWeights())):
                new_weight = self.getLayer(j).getNeuron(ds).getWeights() - \
                             (self.learning_rate * self.getLayer(j).getNeuron(ds).getDeltaSum())
                self.getLayer(j).getNeuron(ds).setWeights(new_weight)
                if not np.array_equal(self.getLayer(j).getNeuron(ds).getWeights(), new_weight):
                    raise Exception("Weights were not saved")

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
    #todo
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
