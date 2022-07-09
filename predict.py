#!/usr/bin/env python3
import ast
import sys
import getopt
import dotenv
from src.operations import Operations

import src.dbops

if __name__ == "__main__":
    inputfile = ''
    values = ''
    argv = sys.argv[1:]
    learning_rate = 1
    dotenv.load_dotenv('.env.testing')
    try:
        opts, args = getopt.getopt(argv, "hi:v:", ["ifile=", "values="])
    except getopt.GetoptError:
        print('predict.py -i <inputfile> -v <values>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('predict.py -i <inputfile> -v <values>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-v", "--values"):
            values = arg
    if (inputfile is None):
        print('the name of the model must be set!')
        exit(1)

    evalueatedVal = ast.literal_eval(values)
    net = src.dbops.DbOps().load_net(inputfile)
    net.setInputs(evalueatedVal)
    neurons = net.getLayer(0).getNeurons()
    for ind, neuron in enumerate(neurons):
        neuron.setValue(evalueatedVal[ind])
        neuron.setSum(evalueatedVal[ind])
    predicted_net = Operations().forwardPropagate(net)

    print("result: ", Operations().get_results(net)[0])
    print("sum: ", net.getLayer(2).getNeuron(0).getSum())

    if Operations().get_results(net)[0] > 0.5:
        print("Activated")
    else:
        print("Not Activated")