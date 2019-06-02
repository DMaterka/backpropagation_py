from backpropagation import Net
from backpropagation import Layer
import numpy as np

# dane testowe oraz wynik jako jeden wiersz tablicy
# reprezentowane przez XOR

numit = 200

test_inputs = np.array([[.05, .1]])
test_outputs = np.array([[.1], [.99]])

# train the network
net = Net(test_inputs, test_outputs, "name")

# set input layer
inputLayer = Layer()
inputLayer.setNeurons(test_inputs, 1)
net.setLayer(0, inputLayer)

# set hidden layer
hiddenLayer = Layer()
hiddenLayer.setNeurons([0, 0])
net.setLayer(1, hiddenLayer)
net.getLayer(1).setWeights([[.15, .2], [.25, 0.3]])

# set output layer
outputLayer = Layer()
outputLayer.setNeurons([0, 0])
net.setLayer(2, outputLayer)
net.getLayer(2).setWeights([[.4, .55], [.5, .55]])

for i in range(1, numit):
    # net.setInputs(inputs[pair])
    net.getLayer(0).setNeurons(test_inputs, 1)
    net.setResults(test_outputs)
    net.forwardPropagate()
    net.backPropagate()

print('results')
net.setInputs(test_inputs[0])
net.forwardPropagate()
