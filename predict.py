#!/usr/bin/env python3
import sys
import getopt
import backpropagation
import pandas as pd
import numpy as np
import sqlite3
import json

if __name__ == "__main__":
    inputfile = ''
    outputfile = ''
    argv = sys.argv[1:]
    learning_rate = 1
    
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])
    except getopt.GetoptError:
        print('predict.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('predict.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
            
    df = pd.read_csv(inputfile)
    
    net = backpropagation.Net(inputfile, [df["col1"], df["col2"]], [df["exp_result"]], int(learning_rate))

    # get model from database
    conn = sqlite3.connect('data/backprop.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM models WHERE name=?', (inputfile,))
    model = c.fetchone()
    c.execute('SELECT * FROM layers WHERE model_id=?', (model['id'],))
    layers = c.fetchall()
    for layer in layers:
        # set a layer
        net_layer = backpropagation.Layer()
        c.execute('SELECT * FROM neurons WHERE layer_id=?', (layer['id'],))
        neurons = c.fetchall()
        for neuron in neurons:
            if layer['layer_index'] > 0:
                net_neuron = backpropagation.Neuron()
                # todo implement storing sums and values of neurons and fetching them back
                net_neuron.setSum([0, 0, 0, 0])
                net_neuron.setValue([0, 0, 0, 0])
                net_layer.setNeuron(net_neuron)
                c.execute('SELECT * FROM neurons WHERE layer_id=?', (layer['id'] - 1,))
                prev_neurons = c.fetchall()
                for prev_neuron in prev_neurons:
                    c.execute('SELECT * FROM weights WHERE neuron_from=? AND neuron_to=?',
                              (prev_neuron['id'], neuron['id']))
                    weight = c.fetchone()
                    net_neuron.setWeights(json.loads(weight['weight']))
        net.setLayer(layer['layer_index'], net_layer)

    # conn.commit()
    conn.close()

    # do the actual prediction
    net.forwardPropagate()

    print("The result is ", net.getLayer(len(net.getLayers()) - 1).getValues())
