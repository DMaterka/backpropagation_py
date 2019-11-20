#!/usr/bin/env python3
import sys
import getopt
from src.backpropagation import Layer, Net, Neuron
import sqlite3
import json
import os
import dotenv


def predict(net: Net):
    inputfile = net.getName()
    if 'testing' in os.environ and os.environ['testing'] == 1:
        dotenv.load_dotenv('.env.testing')
    else:
        dotenv.load_dotenv('.env')
        
    conn = sqlite3.connect('data/' + os.environ['DB_NAME'])
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM models WHERE name=?', (inputfile,))
    model = c.fetchone()
    c.execute('SELECT * FROM layers WHERE model_id=?', (model['id'],))
    layers = c.fetchall()
    for counter, layer in enumerate(layers):
        # set a layer
        if layer['layer_index'] == 0:
            continue
        net_layer = Layer()
        c.execute('SELECT * FROM neurons WHERE layer_id=?', (layer['id'],))
        neurons = c.fetchall()
        for neuron_counter, neuron in enumerate(neurons):
            net_neuron = Neuron()
            net_neuron.setSum(json.loads(neuron['sum']))
            net_neuron.setValue(json.loads(neuron['value']))
            c.execute('SELECT * FROM neurons WHERE layer_id=?', (layer['id'] - 1,))
            prev_neurons = c.fetchall()
            for prev_neuron in prev_neurons:
                c.execute('SELECT * FROM weights WHERE neuron_from=? AND neuron_to=?',
                          (prev_neuron['id'], neuron['id']))
                weight = c.fetchone()
                net_neuron.setWeights(json.loads(weight['weight']), prev_neuron['neuron_index'])
            net_layer.setNeuron(neuron['neuron_index'], net_neuron)
        net.setLayer(layer['layer_index'], net_layer)
    
    conn.close()
    
    # do the actual prediction
    net.forwardPropagate()
    return net


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
            
    net = Net(inputfile, int(learning_rate))
    results = predict(net)
    print("The result is ", results.getLayer(len(results.getLayers()) - 1).getValues())
