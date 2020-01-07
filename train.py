#!/usr/bin/env python3
import sys
import getopt
from src import backpropagation, visualise, dbops
import numpy as np
import ast
import dotenv
import os
import pandas as pd
import re
from operator import itemgetter


def train(inputfile, structure, iterations, learning_rate, batch_size=1):
    if structure != []:
        structure = ast.literal_eval(structure)
    else:
        structure = [3]

    net = backpropagation.Net(inputfile, int(learning_rate))
    
    df = pd.read_csv('data/' + inputfile)
    column_names = df.columns.values
    input_positions = []
    output_positions = []
    for index, value in enumerate(column_names):
        if re.match("input", value):
            input_positions.append(index)
        else:
            output_positions.append(index)

    training_sets = df.iloc[:, :].values
    n_training_sets = []
    #todo make training sets parsing on one go
    for ind, val in enumerate(training_sets):
        init_inputs = itemgetter(input_positions)(val)
        init_outputs = itemgetter(output_positions)(val)
        n_training_sets.insert(ind, [init_inputs, init_outputs])
     
    # prepend list with actual number of inputs
    structure.insert(0, len(input_positions))
    # append list of hidden neurons structure with actual number of outputs
    structure.append(len(output_positions))
    
    for index, value in enumerate(structure):
        layer = backpropagation.Layer()
        layer.setNeurons(np.zeros([value]))
        #todo set weights randomly if none provided
        if index > 0:
            layer.setWeights(np.random.rand(value, structure[index-1]))
        
        if index < len(structure)-1:
            # it seems that different bias values works better
            layer.setBias(np.random.rand(structure[index+1]))
        net.setLayer(index, layer)

    #online training algorithm  with batch size=1 (one training set processed at once)
    learning_curve_data = []
    for i in range(0, int(iterations)):
        modulo = divmod(i, len(n_training_sets))[1]
        inputs, outputs = n_training_sets[modulo]
        net.getLayer(0).setNeurons(inputs.T, False)
        net.setExpectedResults([outputs])
        net.backPropagate()
        total_error = net.calculateTotalError(n_training_sets)
        print(total_error)
        learning_curve_data.append(total_error)
    
    visualise.print_learning_curve(learning_curve_data)
    visualise.print_decision_regions(n_training_sets, net)
    
    total_error = net.calculateTotalError(n_training_sets)
    results = dbops.get_model_results(inputfile)
    
    if results is None:
        dbops.save_net(net, total_error, inputfile)
        print("The model has been saved")
    elif total_error < results['error']:
        dbops.update_net(net, total_error, inputfile)
        print("The model has been updated")
    else:
        print("The total error is the same as previous one")
    
    return net


if __name__ == "__main__":
    # default parameters
    inputfile = 'xor.csv'
    outputfile = ''
    iterations = 1
    argv = sys.argv[1:]
    learning_rate = 1
    structure = []
    batch_size = 1
    
    if 'testing' in os.environ:
        dotenv.load_dotenv('.env.testing')
    else:
        dotenv.load_dotenv('.env')
    if not os.path.isfile('data/' + os.environ['DB_NAME']):
        dbops.createSchema(os.environ['DB_NAME'])
        
    try:
        opts, args = getopt.getopt(argv, "hi:o:n:s:", ["ifile=", "ofile="])
    except getopt.GetoptError:
        print('train.py -i <inputfile> -o <outputfile> -n <number of iterations> -s <hidden layers structure>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('train.py -i <inputfile> -o <outputfile> -n <number of iterations> -s <hidden layers structure>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-n", "--niter"):
            iterations = arg
        elif opt in ("-s", "--struc"):
            structure = arg
            
    net = train(inputfile, structure, iterations, learning_rate, batch_size)
    
    visualise.print_network(net)
    
    print("result is", net.get_results())
