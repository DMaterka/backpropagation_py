#!/usr/bin/env python3
import sys
import getopt
from src import operations, visualise, dbops
from src.DTO.layer import Layer
import numpy as np
import ast
import dotenv
import os
import pandas as pd


def prepare_net(hidden_structure, learning_rate, training_sets, inputfile):
    net = operations.Net(inputfile, float(learning_rate))
    if hidden_structure:
        structure = ast.literal_eval(hidden_structure)
    else:
        structure = [3]
        
    # prepend list with actual number of inputs
    structure.insert(0, len(training_sets[0][0]))
    # append list of hidden neurons structure with actual number of outputs
    structure.append(len(training_sets[0][1]))
    
    for index, value in enumerate(structure):
        layer = Layer()
        layer.setNeurons(np.ones([value]))

        if index > 0:
            layer.setWeights(np.random.rand(value, structure[index - 1]))
        
        if index < len(structure) - 1:
            # it seems that different bias values works better
            layer.setBias(np.random.rand(structure[index + 1]))
        net.setLayer(index, layer)
    return net


def prepare_training_sets(inputfile):
    dotenv.load_dotenv('.env')
    script_path = os.path.dirname(os.path.realpath(__file__))
    if os.getenv('TESTING', False) == True:
        df = pd.read_csv(script_path + '/data/test/' + inputfile)
    else:
        df = pd.read_csv(script_path + '/data/' + inputfile)

    trainingDataCount = int(df.shape[0] * 0.7)
    testingDataCount = int(df.shape[0] * 0.2)
    validationDataCount = int(df.shape[0] * 0.1)

    training_sets = df.iloc[0:trainingDataCount, :].values
    testing_sets = df.iloc[trainingDataCount:trainingDataCount+testingDataCount, :].values
    validation_sets = df.iloc[trainingDataCount+testingDataCount:testingDataCount+trainingDataCount+validationDataCount,:].values

    inputs = df.filter(regex=("input")).values
    outputs = df.filter(regex=("output")).values
    n_training_sets = []
    n_testing_sets = []
    n_validation_sets = []
    # reorder data into sets, do something else to drop the loop #todo
    for ind, val in enumerate(training_sets):
        n_training_sets.insert(ind, [inputs[ind], outputs[ind]])

    for ind, val in enumerate(testing_sets):
        n_testing_sets.insert(ind, [inputs[ind], outputs[ind]])

    for ind, val in enumerate(validation_sets):
        n_validation_sets.insert(ind, [inputs[ind], outputs[ind]])

    return n_training_sets, n_testing_sets, n_validation_sets


def perform_training(net, iterations, training_sets, inputfile, batch_size=1):
    #online training algorithm  with batch size=1 (one training set processed at once)
    net.learning_curve_data = []
    for i in range(0, int(iterations)):
        modulo = divmod(i, len(training_sets))[1]
        inputs, outputs = training_sets[modulo]
        net.getLayer(0).setNeurons(inputs, False)
        net.setExpectedResults(outputs)
        operations.Operations().backPropagate(net)

    return net


def validate_network(net, validation_data):
    ops = operations.Operations()
    valid_count = 0
    for i in range(0, len(validation_data)):
        if ops.predict(net, validation_data[i][0]) == validation_data[i][1]:
            valid_count =+ 1
    return (valid_count/len(validation_data)) * 100


if __name__ == "__main__":
    # default parameters
    inputfile = ''
    outputfile = ''
    iterations = 1
    argv = sys.argv[1:]
    learning_rate = 0.5
    structure = []
    batch_size = 1

    if os.getenv('TESTING', False) == True:
        project_dotenv = '.env.testing'
    else:
        project_dotenv = '.env'

    dotenv.load_dotenv(dotenv_path=project_dotenv)

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

    # DO something with testing sets
    training_sets, testing_sets, validation_sets = prepare_training_sets(inputfile)
    net = prepare_net(structure, learning_rate, training_sets, inputfile)
    trained_net = perform_training(net, iterations, training_sets, inputfile, batch_size)

    valid_percent = validate_network(net, validation_sets)

    print("Valid results number is equal to" + str(valid_percent))

    visualise.print_learning_curve(net.learning_curve_data)
    visualise.print_decision_regions(training_sets, net)

    total_error = operations.Operations().calculateTotalError(net)

    dbOpsObject = dbops.DbOps(inputfile)
    results = dbOpsObject.get_model_results(inputfile)

    if results is None:
        dbOpsObject.save_net(net, total_error, inputfile)
        print("The model has been saved")
    elif total_error < results['error']:
        dbOpsObject.update_net(net, total_error, inputfile)
        print(
            "Total error is smaller or equal to the previous one "
            + str(total_error) + " < " + str(results['error'])
        )
        print("The model has been updated")
    else:
        print(
            "Total error is not smaller or equal to the previous one "
            + str(total_error) + " > " + str(results['error'])
        )

    visualise.print_network(trained_net)
    
    print("result is", operations.Operations().get_results(trained_net))
