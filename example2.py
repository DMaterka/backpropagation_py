from backpropagation import Net
from backpropagation import Layer
"""dane testowe oraz wynik jako jeden wiersz tablicy reprezentowane przez XOR
@TODO small differences of -4 order
"""
numit = 3

test_inputs = [[.05, .1]]
test_outputs = [[.01], [.99]]

# train the network
net = Net("name", test_inputs, test_outputs, 0.5)

# set input layer
inputLayer = Layer()
inputLayer.setNeurons(test_inputs, 1)
inputLayer.setBias([.35, .35])
net.setLayer(0, inputLayer)

# set hidden layer
hiddenLayer = Layer()
hiddenLayer.setNeurons([[0, 0]])
hiddenLayer.setBias([.6, .6])
hiddenLayer.setWeights([[.15, .20], [.25, .3]])
net.setLayer(1, hiddenLayer)

# set output layer
outputLayer = Layer()
outputLayer.setNeurons([[0, 0]])
outputLayer.setWeights([[.4, .45], [.50, .55]])
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
