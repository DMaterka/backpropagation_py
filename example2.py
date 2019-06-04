from backpropagation import Net
from backpropagation import Layer
import numpy as np
#@TODO break second n-th layer bias connection to previous layer
# dane testowe oraz wynik jako jeden wiersz tablicy
# reprezentowane przez XOR

numit = 200

test_inputs = np.array([[.05, .1]])
test_outputs = np.array([[.1], [.99]])

# train the network
net = Net("name", test_inputs, test_outputs)

# set input layer
inputLayer = Layer()
inputLayer.setNeurons(test_inputs, 1)
inputLayer.setBias()
net.setLayer(0, inputLayer)

# set hidden layer
hiddenLayer = Layer()
hiddenLayer.setNeurons([0, 0])
hiddenLayer.setBias()
hiddenLayer.setWeights([[.15, .25, .35], [.2, .3, .35]])
net.setLayer(1, hiddenLayer)

# set output layer
outputLayer = Layer()
outputLayer.setNeurons([0, 0])
outputLayer.setWeights([[.4, .5, .6], [.45, .55, .6]])
net.setLayer(2, outputLayer)

for i in range(1, numit):
    # net.setInputs(inputs[pair])
    # net.getLayer(0).setNeurons(test_inputs, 1)
    # net.setResults(test_outputs)
    net.forwardPropagate()
    net.backPropagate()

print('results')
net.setInputs(test_inputs[0])
net.forwardPropagate()
