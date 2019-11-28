#!/usr/bin/env python3
import sys
import getopt
from src import backpropagation, db_create
import numpy as np
import ast
import sqlite3
import json
import os
import re
import matplotlib.pyplot as plt


def train1d(net: backpropagation.Net, structure, iterations):
    # define number of layers and neurons
    inputfile = net.getName()
    # input layer already set up
    if structure != []:
        structure = ast.literal_eval(structure)
    else:
        structure = [3, 1]
    
    for index in range(0, len(structure)):
        layer = backpropagation.Layer()
        layer.setNeurons(np.zeros([structure[index], 1]))
        layer.setWeights(np.random.random_integers(1, 99, [structure[index], len(net.getLayer(index).getNeurons())])/100)
        if index != len(structure):
            layer.setBias(np.random.rand(len(net.getLayer(index).getNeurons())))
        net.setLayer(index + 1, layer)
    
    curve = []
    mini_batch = []
    mini_batch_size = 12
    for i in range(0, int(iterations) * mini_batch_size):
        inp1 = np.random.randint(1, 99) / 100
        inp2 = np.random.randint(1, 99) / 100
        result = np.bitwise_xor(round(inp1), round(inp2))
        
        if result == 0: result += 0.01
        if result == 1: result -= 0.01
        mini_batch.insert(i, np.array([[inp1], [inp2], [result]]))
    i = 0
    j = 0
    while j < int(iterations)*mini_batch_size:
        print("epoch " + str(j) + " out of " + str(int(iterations) * mini_batch_size), end='\r')
        mini_batch_inputs = np.array(mini_batch[i*mini_batch_size:i*mini_batch_size+mini_batch_size]).T[0][0:2]
        mini_batch_results = np.array(mini_batch[i*mini_batch_size:i*mini_batch_size+mini_batch_size]).T[0][2]
        net.getLayer(0).setNeurons(mini_batch_inputs, 1)
        net.setExpectedResults(np.array([mini_batch_results]))
        net.forwardPropagate()
        net.backPropagate()
        # show batch error instead
        curve.append(np.mean(abs(net.error)))
        j = j + 1
            
    # print learning curve
    plt.plot(range(0, j), curve)
    plt.tight_layout()
    plt.show()
    
    net.getLayer(0).setNeurons([[0.01], [0.01]], 1)
    net.forwardPropagate()
    print("The result is ", net.getLayer(len(net.getLayers()) - 1).getValues())
    
    net.getLayer(0).setNeurons([[0.99], [0.01]], 1)
    net.forwardPropagate()
    print("The result is ", net.getLayer(len(net.getLayers()) - 1).getValues())
    
    net.getLayer(0).setNeurons([[0.01], [0.99]], 1)
    net.forwardPropagate()
    print("The result is ", net.getLayer(len(net.getLayers()) - 1).getValues())
    
    net.getLayer(0).setNeurons([[0.99], [0.99]], 1)
    net.forwardPropagate()
    print("The result is ", net.getLayer(len(net.getLayers()) - 1).getValues())
    
    dbname = re.sub("\..*", "", inputfile) + '.db'
    conn = sqlite3.connect('data/' + dbname)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM models WHERE name=?', (inputfile,))
    results = c.fetchone()
    total_error = np.average(np.abs(net.error))
    
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
    learning_rate = 1
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
    
    net = train1d(net, structure, iterations)
    
    net.print_network()
    
    print("result is", net.getLayer(len(net.getLayers()) - 1).getValues())
    print("error is", net.error)
    print("expected result is", net.getExpectedResults())
