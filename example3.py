from backpropagation import Net
from backpropagation import Layer
import numpy as np
#@TODO
numit = 2
inputs = np.array([[0.01, 0.99], [0.01, 0.01], [0.99, 0.99], [0.99, 0.01]])
results = np.array([[0.99], [0.01], [0.01], [0.99]])

# inputs = np.array([[0.99, 0.99]])
# results = np.array([[0.01]])
test_inputs = np.array([[.05, .1]])
test_outputs = np.array([[.1, .99]])
# train the network
net = Net(test_inputs, test_outputs, "name")
# set input layer
inputLayer = Layer()
inputLayer.setNeurons(test_inputs[0], 1)
net.setLayer(0, inputLayer)
# set hidden layer
hiddenLayer = Layer()
hiddenLayer.setNeurons([0, 0])
net.setLayer(1, hiddenLayer)
net.getLayer(1).setWeights([[.15, .2], [.25, 0.3]])

# set second hidden layer
hiddenLayer1 = Layer()
hiddenLayer1.setNeurons([0, 0])
net.setLayer(2, hiddenLayer1)
net.getLayer(2).setWeights([[.4, .55], [.5, .55]])

hiddenLayer2 = Layer()
hiddenLayer2.setNeurons([0, 0, 0, 0, 0])
net.setLayer(3, hiddenLayer2)
net.getLayer(3).setWeights([[0.6, 0.8, 0.3, 0.7, 0.2]])

# set output layer
# outputLayer = Layer()
# outputLayer.setNeurons([0])
# net.setLayer(3, outputLayer)
# net.getLayer(3).setWeights([[.3, .5, .9, .7]])
# net.setWeights()
t = len(inputs)
for i in range(1, numit):
    # net.setInputs(inputs[pair])
    net.getLayer(0).setNeurons(test_inputs[0], 1)
    net.setResults(test_outputs)
    net.forwardPropagate()
    net.backPropagate()

print('results')
net.setInputs(inputs[0])
net.forwardPropagate()

net.setInputs(inputs[1])
net.forwardPropagate()

net.setInputs(inputs[2])
net.forwardPropagate()

net.setInputs(inputs[3])
net.forwardPropagate()
# validate the network