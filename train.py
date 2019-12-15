#!/usr/bin/env python3
import sys
import getopt
from src import backpropagation
import numpy as np
import ast
import sqlite3
import json
import dotenv
import os
import pandas as pd
import re
from operator import itemgetter
import matplotlib.pyplot as plt


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
        layer.setWeights(np.random.rand(value, value))
        # it seems that different bias values works well
        layer.setBias(np.random.rand(value))
        net.setLayer(index, layer)

    #online training algorithm  with batch size=1 (one training set processed at once)
    curve = []
    for i in range(0, int(iterations)):
        modulo = divmod(i, len(n_training_sets))[1]
        inputs, outputs = n_training_sets[modulo]
        net.getLayer(0).setNeurons(inputs.T)
        net.setExpectedResults([outputs])
        net.backPropagate()
        total_error = net.calculateTotalError(n_training_sets)
        print(total_error)
        curve.append(total_error)

    plt.plot(range(0, int(iterations)), curve)
    plt.tight_layout()
    plt.show()
    
    if 'testing' in os.environ:
        dotenv.load_dotenv('.env.testing')
    else:
        dotenv.load_dotenv('.env')
        
    conn = sqlite3.connect('data/' + os.environ['DB_NAME'])
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM models WHERE name=?', (inputfile,))
    results = c.fetchone()

    total_error = net.calculateTotalError(n_training_sets)
    
    if results is None:
        save_net(net, total_error)
    elif total_error < results['error']:
        update_net(net, total_error)
    else:
        print("The total error is the same as previous one")
    
    conn.commit()
    conn.close()
    return net


def save_net(net: backpropagation.Net, total_error):
    conn = sqlite3.connect('data/' + os.environ['DB_NAME'])
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('INSERT INTO models (name, error) VALUES (?, ?)', (inputfile, total_error))
    modelid = c.lastrowid
    for layer_index in range(0, len(net.getLayers())):
        c.execute('INSERT INTO layers (layer_index, model_id) VALUES (?, ?)', (layer_index, modelid))
        layerid = c.lastrowid
        for neuron_index in range(0, len(net.getLayer(layer_index).getNeurons())):
            sums = net.getLayer(layer_index).getNeuron(neuron_index).getSum()
            values = net.getLayer(layer_index).getNeuron(neuron_index).getValue()
            c.execute(
                'INSERT INTO neurons (neuron_index, layer_id, sum, value) VALUES (?, ?, ?, ?)', (
                    neuron_index, layerid, json.dumps(sums.tolist()), json.dumps(values.tolist())
                )
            )
            neuron_id = c.lastrowid
            if layer_index > 0:
                for prev_layer_neuron_index in range(0, len(net.getLayer(layer_index - 1).getNeurons())):
                    weights = net.getLayer(layer_index).getNeuron(neuron_index).getWeights()[
                        prev_layer_neuron_index]
                    c.execute('SELECT * FROM neurons WHERE neuron_index=? AND layer_id=?',
                              (prev_layer_neuron_index, layerid - 1))
                    prev_neuron = c.fetchone()
                    c.execute('INSERT INTO weights (neuron_from, neuron_to, weight) VALUES (?, ?, ?)',
                              (prev_neuron['id'], neuron_id, json.dumps(weights.tolist())))

def update_net(net: backpropagation.Net, total_error):
    conn = sqlite3.connect('data/' + os.environ['DB_NAME'])
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM models WHERE name=?', (inputfile,))
    results = c.fetchone()
    modelid = results['id']
    c.execute('UPDATE models SET error=? WHERE id=?', (total_error, modelid))
    for layer_index in range(0, len(net.getLayers())):
        c.execute('SELECT * FROM layers WHERE model_id=? AND layer_index=?', (modelid, layer_index))
        layer = c.fetchone()
        for neuron_index in range(0, len(net.getLayer(layer_index).getNeurons())):
            c.execute('SELECT * FROM neurons WHERE layer_id=? AND neuron_index=?', (layer['id'], neuron_index))
            neuron = c.fetchone()
            if layer_index > 0:
                for prev_layer_neuron_index in range(0, len(net.getLayer(layer_index - 1).getNeurons())):
                    weights = net.getLayer(layer_index).getNeuron(neuron_index).getWeights()[
                        prev_layer_neuron_index]
                    c.execute('SELECT * FROM neurons WHERE neuron_index=? AND layer_id=?',
                              (prev_layer_neuron_index, layer['id'] - 1))
                    prev_neuron = c.fetchone()
                    c.execute('UPDATE weights SET weight=? WHERE neuron_from=? AND neuron_to=?',
                              (json.dumps(weights.tolist()), prev_neuron['id'], neuron['id']))


if __name__ == "__main__":
    inputfile = ''
    outputfile = ''
    iterations = 1
    argv = sys.argv[1:]
    learning_rate = 1
    structure = []
    batch_size = 1
    
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
    
    net.print_network()
    
    print("result is", net.get_results())
