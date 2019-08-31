#!/usr/bin/env python3
import sys
import getopt
import backpropagation
import csv

if __name__ == "__main__":
    print('Number of arguments:', len(sys.argv), 'arguments.')
    print('Argument List:', str(sys.argv))
    inputfile = ''
    outputfile = ''
    iterations = 0
    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])
    except getopt.GetoptError:
        print('train.py -i <inputfile> -o <outputfile> -n <number of iterations>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('train.py -i <inputfile> -o <outputfile> -n <number of iterations>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-n", "--niter"):
            iterations = arg
            
    print('Input file is "', inputfile)
    print('Output file is "', outputfile)

    csv = csv.reader(inputfile)

    net = backpropagation.Net("nameToDO", [[0.99], [0.99]], [[0.01]])

    # define number of layers and neurons

    for i in range(1, iterations):
        # net.getLayer(0).setNeurons(np.array([[0.99, 0.99]]), 1)
        # net.setResults(np.array([[0.01]]))
        net.forwardPropagate()
        # net.print_network()
        net.backPropagate()
        
    print("result is", net.getLayer(len(net.getLayers())).getValues())
