#!/usr/bin/env python3
import sys
import getopt
import backpropagation
import pandas as pd
import numpy as np
import ast

if __name__ == "__main__":
    inputfile = ''
    outputfile = ''
    iterations = 1
    argv = sys.argv[1:]
    learning_rate = 1
    structure = []
    
    try:
        opts, args = getopt.getopt(argv, "hi:o:n:s:", ["ifile=", "ofile="])
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
    
    net = backpropagation.Net("nameToDO", [df["col1"], df["col2"]], [df["exp_result"]], int(learning_rate))
    
    # define number of layers and neurons
    
    # input layer already set up
    if structure != []:
        structure = ast.literal_eval(structure)
    else:
        structure = [3, 1]
        
    for index in range(0, len(structure)):
        layer = backpropagation.Layer()
        layer.setNeurons(np.zeros([structure[index], 1]))
        layer.setWeights(np.random.rand(structure[index], len(net.getLayer(index).getNeurons())))
        net.setLayer(index + 1, layer)

    for i in range(0, int(iterations)):
        # net.getLayer(0).setNeurons(np.array([[0.99, 0.99]]), 1)
        # net.setResults(np.array([[0.01]]))
        net.forwardPropagate()
        # net.print_network()
        net.backPropagate()
        
    print("result is", net.getLayer(len(net.getLayers())-1).getValues())
    print("error is", net.error)
    # save model to database