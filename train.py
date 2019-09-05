#!/usr/bin/env python3
import sys
import getopt
import backpropagation
import pandas as pd
import numpy as np
import ast
import sqlite3

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
    
    net = backpropagation.Net(inputfile, [df["col1"], df["col2"]], [df["exp_result"]], int(learning_rate))
    
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
        
    # save model to database
    conn = sqlite3.connect('data/backprop.db')
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
            for neuron_index in range(0, len(net.getLayer(layer_index).getNeurons())-1):
                c.execute('INSERT INTO neurons (neuron_index, layer_id) VALUES (?, ?)', (neuron_index, layerid))
                if layer_index > 0:
                    for weight_index in range(0, len(net.getLayer(layer_index).getNeuron(neuron_index).getWeights())):
                        for prev_layer_neuron_index in range(0, len(net.getLayer(layer_index-1).getNeurons())):
                            weight = net.getLayer(layer_index).getNeuron(neuron_index).getWeights()[neuron_index][prev_layer_neuron_index]
                            c.execute('INSERT INTO weights (neuron_from, neuron_to, weight) VALUES (?, ?, ?)',
                                      (prev_layer_neuron_index, weight_index, weight))
    elif total_error < results['error']:
        c.execute('UPDATE models SET error=? WHERE id=?', (total_error, results['id']))
        c.execute('SELECT * FROM layers WHERE model_id=?', (results['id']))
        # TODO update weights if error is lower than existing
        c.execute('SELECT * FROM weights WHERE layers=?', (results['id']))
        
    conn.commit()
    conn.close()

    print("result is", net.getLayer(len(net.getLayers()) - 1).getValues())
    print("error is", total_error)
