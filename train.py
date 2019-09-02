#!/usr/bin/env python3
import sys
import getopt
import backpropagation
import pandas as pd

if __name__ == "__main__":
    inputfile = ''
    outputfile = ''
    iterations = 1
    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv, "hi:o:n:", ["ifile=", "ofile="])
    except getopt.GetoptError:
        print('train.py -i <inputfile> -o <outputfile> -n <number of iterations> -s <network structure>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('train.py -i <inputfile> -o <outputfile> -n <number of iterations> -s <network structure>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-n", "--niter"):
            iterations = arg
        elif opt in ("-s", "--struc"):
            structure = arg

    df = pd.read_csv(inputfile)
    
    net = backpropagation.Net("nameToDO", [df["col1"], df["col2"]], [df["exp_result"]])
    
    # define number of layers and neurons
    
    # input layer already set up

    # set hidden layer
    hiddenLayer = backpropagation.Layer()
    hiddenLayer.setNeurons([[0], [0], [0]])
    hiddenLayer.setWeights([[.3, .5], [.4, .9], [.8, .2]])
    net.setLayer(1, hiddenLayer)

    # set second hidden layer
    outputLayer = backpropagation.Layer()
    outputLayer.setNeurons([[0]])
    net.setLayer(2, outputLayer)
    net.getLayer(2).setWeights([[.3, .5, .9]])

    for i in range(0, int(iterations)):
        # net.getLayer(0).setNeurons(np.array([[0.99, 0.99]]), 1)
        # net.setResults(np.array([[0.01]]))
        net.forwardPropagate()
        # net.print_network()
        net.backPropagate()
        
    print("result is", net.getLayer(len(net.getLayers())-1).getValues())
