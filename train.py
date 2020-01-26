#!/usr/bin/env python3
import sys
import getopt
from src import backpropagation, visualise, dbops
import numpy as np
import ast
import dotenv
import os
import pandas as pd


def prepare_net(hidden_structure, learning_rate, training_sets, inputfile):
    net = backpropagation.Net(inputfile, float(learning_rate))
    if hidden_structure:
        structure = ast.literal_eval(hidden_structure)
    else:
        structure = [3]
        
    # prepend list with actual number of inputs
    structure.insert(0, len(training_sets[0][0]))
    # append list of hidden neurons structure with actual number of outputs
    structure.append(len(training_sets[0][1]))
    
    for index, value in enumerate(structure):
        layer = backpropagation.Layer()
        layer.setNeurons(np.zeros([value]))
        # todo set weights randomly if none provided
        if index > 0:
            layer.setWeights(np.random.rand(value, structure[index - 1]))
        
        if index < len(structure) - 1:
            # it seems that different bias values works better
            layer.setBias(np.random.rand(structure[index + 1]))
        net.setLayer(index, layer)
    return net


def prepare_training_sets(inputfile):
    if 'testing' in os.environ:
        df = pd.read_csv('test/data/' + inputfile)
    else:
        df = pd.read_csv('data/' + inputfile)
    
    training_sets = df.iloc[:, :].values
    inputs = df.filter(regex=("input")).values
    outputs = df.filter(regex=("output")).values
    n_training_sets = []
    # reorder data into sets, do something else to drop the loop #todo
    for ind, val in enumerate(training_sets):
        n_training_sets.insert(ind, [inputs[ind], outputs[ind]])
     
    return n_training_sets


def perform_training(net, iterations, training_sets, inputfile, batch_size=1):
    #online training algorithm  with batch size=1 (one training set processed at once)
    learning_curve_data = []
    for i in range(0, int(iterations)):
        modulo = divmod(i, len(training_sets))[1]
        inputs, outputs = training_sets[modulo]
        net.getLayer(0).setNeurons(inputs.T, False)
        net.setExpectedResults([outputs])
        net.backPropagate()
        total_error = net.calculateTotalError(training_sets)
        print(total_error)
        learning_curve_data.append(total_error)
    
    visualise.print_learning_curve(learning_curve_data)
    visualise.print_decision_regions(training_sets, net)
    
    total_error = net.calculateTotalError(training_sets)
    results = dbops.get_model_results(inputfile)
    
    if results is None:
        dbops.save_net(net, total_error, inputfile)
        print("The model has been saved")
    elif total_error < results['error']:
        dbops.update_net(net, total_error, inputfile)
        print("The model has been updated")
    else:
        print("Total error is the same as previous one")
    
    return net


if __name__ == "__main__":
    # default parameters
    inputfile = 'xor.csv'
    outputfile = ''
    iterations = 1
    argv = sys.argv[1:]
    learning_rate = 0.5
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
    
    training_sets = prepare_training_sets(inputfile)
    net = prepare_net(structure, learning_rate, training_sets, inputfile)
    trained_net = perform_training(net, iterations, training_sets, inputfile, batch_size)
    
    visualise.print_network(trained_net)
    
    print("result is", trained_net.get_results())
