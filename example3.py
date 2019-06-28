from backpropagation import Net
from backpropagation import Layer
from datetime import datetime
#@TODO
numit = 1000
inputs = [[.01], [.99], [.01], [.01], [.99], [.99], [.99], [.01]]
results = [[.99], [.01], [.01], [.99]]

# train the network
net = Net("name", inputs, results, 0.3)
# set input layer
inputLayer = Layer()
inputLayer.setNeurons(inputs, 1)
inputLayer.setBias([.4, .6])
net.setLayer(0, inputLayer)
# set hidden layer
hiddenLayer = Layer()
hiddenLayer.setNeurons([[0], [0]])
hiddenLayer.setBias([.4, .6, .5])
hiddenLayer.setWeights([[.15, .2, .1, .6, .2, .9, .5, .3], [.15, .2, .1, .6, .2, .9, .5, .3]])
net.setLayer(1, hiddenLayer)

# set second hidden layer
hiddenLayer1 = Layer()
hiddenLayer1.setNeurons([[0], [0], [0]])
hiddenLayer1.setBias([.2, .7, .9, .3, .8])
hiddenLayer1.setWeights([[.4, .55], [.5, .55], [.5, .55]])
net.setLayer(2, hiddenLayer1)

hiddenLayer2 = Layer()
hiddenLayer2.setNeurons([[0], [0], [0], [0], [0]])
hiddenLayer2.setBias([.5, .2, .6, .8])
hiddenLayer2.setWeights([[0.6, 0.8, 0.3], [0.6, 0.8, 0.3], [0.6, 0.8, 0.3], [0.6, 0.8, 0.3], [0.6, 0.8, 0.3]])
net.setLayer(3, hiddenLayer2)

# set output layer
outputLayer = Layer()
outputLayer.setNeurons([[0], [0], [0], [0]])
outputLayer.setWeights([[.3, .5, .9, .7, .5], [.3, .5, .9, .7, .5], [.3, .5, .9, .7, .5], [.3, .5, .9, .7, .5]])
net.setLayer(4, outputLayer)

# net.setWeights()
net.setResults(results)
timeNow = datetime.now()
for i in range(1, numit):
    net.forwardPropagate()
    net.backPropagate()
duration = datetime.now() - timeNow
print('network trained in ' + str(duration.total_seconds()) + ' seconds')
net.forwardPropagate()
# validate the network