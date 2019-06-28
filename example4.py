from backpropagation import Net
from backpropagation import Layer

#@TODO doeasn't work yet
numit = 100
inputs = [[[.01, .99], [.99, .01]], [[.01, .01], [.99, .99]]]
results = [[[.99, .99], [.01, .01]], [[.99, .99], [.01, .01]]]

# train the network
net = Net(inputs, results, "name", .3)
# set input layer
inputLayer = Layer()
inputLayer.setNeurons(inputs, 1)
net.setLayer(0, inputLayer)
# set hidden layer
hiddenLayer = Layer()
hiddenLayer.setNeurons([[[0, 0], [0, 0]], [[0, 0], [0, 0]]])
net.setLayer(1, hiddenLayer)
net.getLayer(1).setWeights([[[[.4, .55], [.4, .55]], [[.4, .55], [.4, .55]], [[.4, .55], [.4, .55]], [[.4, .55], [.4, .55]], [[.4, .55], [.4, .55]], [[.4, .55], [.4, .55]], [[.4, .55], [.4, .55]], [[.4, .55], [.4, .55]]],
                            [[[.4, .55], [.4, .55]], [[.4, .55], [.4, .55]], [[.4, .55], [.4, .55]], [[.4, .55], [.4, .55]], [[.4, .55], [.4, .55]], [[.4, .55], [.4, .55]], [[.4, .55], [.4, .55]], [[.4, .55], [.4, .55]]]])

# set second hidden layer
hiddenLayer1 = Layer()
hiddenLayer1.setNeurons([[[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]])
net.setLayer(2, hiddenLayer1)
net.getLayer(2).setWeights([[[[.1, .55], [.4, .55]], [[.4, .55], [.4, .55]]],
                            [[[.2, .55], [.4, .55]], [[.4, .55], [.4, .55]]],
                            [[[.3, .55], [.4, .55]], [[.4, .55], [.4, .55]]]])

hiddenLayer2 = Layer()
hiddenLayer2.setNeurons([[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]])
net.setLayer(3, hiddenLayer2)
net.getLayer(3).setWeights([[[.8, .2], [.5, .9]], [[.2, .6], [.2, .5]], [[.7, .8], [.4, .5]]])

# set output layer
outputLayer = Layer()
outputLayer.setNeurons([[0, 0], [0, 0]])
net.setLayer(4, outputLayer)
net.getLayer(4).setWeights([[[.15, .2, 0.6, 0.8, 0.3], [.4, .8, .15, .2, .2]], [[0.7, 0.2, .87, .4, .9], [0.3, 0.7, .4, .4, .3]]])
# net.setWeights()
net.setResults(results)
for i in range(1, numit):
    net.forwardPropagate()
    net.backPropagate()

print('results')
net.forwardPropagate()
# validate the network