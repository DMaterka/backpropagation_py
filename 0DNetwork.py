from backpropagation import Net
from backpropagation import Layer

# dane testowe oraz wynik jako jeden wiersz tablicy
# reprezentowane przez XOR

#parameters
numit = 3
inputs = [[0.01, 0.99], [0.01, 0.01], [0.99, 0.99], [0.99, 0.01]]
results = [[0.99], [0.01], [0.01], [0.99]]

# train the network
net = Net("0dnet", [[0.99], [0.99]], [[0.01]])

# set hidden layer
hiddenLayer = Layer()
hiddenLayer.setNeurons([[0], [0], [0]])
hiddenLayer.setWeights([[.3, .5], [.4, .9], [.8, .2]])
net.setLayer(1, hiddenLayer)

# set second hidden layer
outputLayer = Layer()
outputLayer.setNeurons([[0]])
net.setLayer(2, outputLayer)
net.getLayer(2).setWeights([[.3, .5, .9]])

# net.setWeights()
for i in range(1, numit):
    # net.getLayer(0).setNeurons(np.array([[0.99, 0.99]]), 1)
    # net.setResults(np.array([[0.01]]))
    net.forwardPropagate()
    # net.print_network()
    net.backPropagate()

print(net.getLayer(len(net.getLayers())-1).getValues())

# #store the network
# conn = sqlite3.connect(cfg.DbConfig.DIR + cfg.DbConfig.NAME)
#
# conn.close
# validate the network