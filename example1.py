from backpropagation import Net
from backpropagation import Layer
import numpy as np

# dane testowe oraz wynik jako jeden wiersz tablicy
# reprezentowane przez XOR

numit = 2
inputs = np.array([[0.01, 0.99], [0.01, 0.01], [0.99, 0.99], [0.99, 0.01]])
results = np.array([[0.99], [0.01], [0.01], [0.99]])

# train the network
net = Net(np.array([[0.99, 0.99]]), np.array([[0.01]]), "name")

# set input layer
inputLayer = Layer()
inputLayer.setNeurons(np.array([[0.99, 0.99]]), 1)
net.setLayer(0, inputLayer)

# set hidden layer
hiddenLayer = Layer()
hiddenLayer.setNeurons([0, 0, 0])
net.setLayer(1, hiddenLayer)
net.getLayer(1).setWeights([[.8, .2], [.4, .9], [.3, .5]])

# set second hidden layer
outputLayer = Layer()
outputLayer.setNeurons([0])
net.setLayer(2, outputLayer)
net.getLayer(2).setWeights([[.3, .5, .9]])

# net.setWeights()
for i in range(1, numit):
    net.getLayer(0).setNeurons(np.array([[0.99, 0.99]]), 1)
    net.setResults(np.array([[0.01]]))
    net.forwardPropagate()
    net.backPropagate()

print('expected results: [[0.77381515]] and [[0.69451532]]')
net.setInputs(inputs[2])
net.forwardPropagate()

# validate the network