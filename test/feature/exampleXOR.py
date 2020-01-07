#!/usr/bin/env python3
import sys
import getopt
from src import backpropagation, visualise, dbops
import dotenv
import os

def train(net: backpropagation.Net, iterations):
    # define number of layers and neurons
    inputfile = "xor.csv"
    # input layer already set up
    
    structure = [2, 2]
    inputLayer = backpropagation.Layer()
    inputLayer = inputLayer.setNeurons([[0], [0]])
    inputLayer.setBias([.35, .35])
    net.setLayer(1, inputLayer)
    
    hiddenLayer = backpropagation.Layer()
    hiddenLayer.setNeurons([[0], [0]])
    hiddenLayer.setWeights([[.15, .2], [.25, .3]])
    hiddenLayer.setBias([.6, .6])
    net.setLayer(1, hiddenLayer)
    outputLayer = backpropagation.Layer()
    outputLayer.setNeurons([[0], [0]])
    outputLayer.setWeights([[.4, .45], [.5, .55]])
    net.setLayer(2, outputLayer)
    net.setExpectedResults([[0], [0]])
    
    learning_curve_data = []
    training_sets = [
        [[[0], [0]], [[0], [0]]],
        [[[0], [1]], [[1], [1]]],
        [[[1], [0]], [[1], [1]]],
        [[[1], [1]], [[0], [0]]]
    ]
    for iteration in range(0, int(iterations)):
        # init, expected = random.choice(training_sets)
        modulo = divmod(iteration, 4)[1]
        init, expected = training_sets[modulo]
        net.getLayer(0).setNeurons(init)
        net.setExpectedResults(expected)
        net.backPropagate()
        total_error = net.calculateTotalError(training_sets)
        print(total_error)
        learning_curve_data.append(total_error)
    visualise.print_learning_curve(learning_curve_data)
    visualise.print_decision_regions(training_sets, net)
    print(net.calculateTotalError(training_sets))
    
    net.getLayer(0).setNeurons([[0], [0]])
    net.forwardPropagate()
    print(net.get_results())
    
    results = dbops.get_model_results(inputfile)

    if results is None:
        dbops.save_net(net, total_error, inputfile)
        print("The model has been saved")
    elif total_error < results['error']:
        dbops.update_net(net, total_error, inputfile)
        print("The model has been updated")
    else:
        print("The total error is the same as previous one")
    
    testnet = dbops.load_net(inputfile)
    testnet.forwardPropagate()
    # TODO must be equal
    print(testnet.get_results())
    print(net.get_results())
    visualise.print_network(testnet)
    return net


if __name__ == "__main__":
    inputfile = ''
    outputfile = ''
    iterations = 1
    argv = sys.argv[1:]
    learning_rate = 0.5
    structure = []
    results = []
    
    try:
        opts, args = getopt.getopt(argv, "hi:o:n:s:", ["ifile=", "ofile="])
    except getopt.GetoptError:
        print('train1d.py -i <inputfile> -o <outputfile> -n <number of iterations> -s <network structure>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('train1d.py -i <inputfile> -o <outputfile> -n <number of iterations> -s <network structure>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-n", "--niter"):
            iterations = arg
        elif opt in ("-s", "--struc"):
            structure = arg
        elif opt in ("-r", "--results"):
            results = arg
    
    net = backpropagation.Net(inputfile, learning_rate)
    
    # dotenv.load_dotenv('.env')
    dotenv.load_dotenv('.env.testing')
    if not os.path.isfile('data/' + os.environ['DB_NAME']):
        dbops.createSchema(os.environ['DB_NAME'])
    net = train(net, iterations)
    visualise.print_network(net)
