#!/usr/bin/env python3
import sys
import getopt
from src import backpropagation, db_create
import ast
import sqlite3
import json
import os
import re
import matplotlib.pyplot as plt

def train(net: backpropagation.Net, structure, iterations):
    # define number of layers and neurons
    inputfile = net.getName()
    # input layer already set up
    if structure != []:
        structure = ast.literal_eval(structure)
    else:
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
    
    curve = []
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
        curve.append(total_error)
    plt.plot(range(0, int(iterations)), curve)
    print(net.calculateTotalError(training_sets))
    plt.tight_layout()
    plt.show()
    net.getLayer(0).setNeurons([[0], [0]])
    net.forwardPropagate()
    print(net.get_results())
    dbname = re.sub("\..*", "", inputfile) + '.db'
    conn = sqlite3.connect('data/' + dbname)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM models WHERE name=?', (inputfile,))
    results = c.fetchone()
    
    if results is None:
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
    elif total_error < results['error']:
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
    else:
        print("The total error is the same as previous one")
    
    conn.commit()
    conn.close()
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
    
    dbname = re.sub("\..*", "", inputfile) + '.db'
    
    if not os.path.isfile('data/' + dbname):
        db_create.createSchema(dbname)

    net = train(net, structure, iterations)
    net.print_network()
