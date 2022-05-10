#!/usr/bin/env python3
import sys
import getopt
from src.backpropagation import Layer, Net, Neuron
import sqlite3
import os
import dotenv


def predict(net: Net, values):
    inputfile = net.getName()
    dotenv.load_dotenv('.env')
    if os.environ['TESTING']:
        dotenv.load_dotenv('.env.testing')
        dbname = 'data/test/' + os.environ['DB_NAME']
    else:
        dotenv.load_dotenv('.env')
        dbname = 'data/' + os.environ['DB_NAME']
        
    conn = sqlite3.connect(dbname)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM models WHERE name=?', (inputfile,))
    model = c.fetchone()

    c.execute('SELECT * FROM layers WHERE model_id=? ORDER BY id', (model['id'],))
    layers = c.fetchall()
    for layer_counter, layer in enumerate(layers):
        net_layer = Layer()

        c.execute('SELECT * FROM neurons WHERE layer_id=?', (layer['id'],))
        neurons = c.fetchall()

        for neuron_counter, neuron in enumerate(neurons):
            net_neuron = Neuron()

            if neuron['is_bias']:
                net_neuron.is_bias = True
            else:
                net_neuron.setSum(neuron['sum'])
                net_neuron.setValue(neuron['value'])

            if layer_counter > 0:
                c.execute('SELECT id FROM neurons WHERE layer_id=? ',
                          (layers[layer_counter-1]['id'],))
                c.row_factory = single_result_factory
                prev_layer_neurons = c.fetchall()
                c.row_factory = sqlite3.Row
                converted_list = [str(element) for element in prev_layer_neurons]
                c.execute("SELECT weight FROM weights WHERE neuron_from IN (" + ",".join(converted_list) + ") AND neuron_to=?",
                          (neuron['id'],))
                weights = c.fetchall()
                for weight_ind, weight in enumerate(weights):
                    net_neuron.setWeights(weight['weight'], weight_ind)
            else:
                if neuron['is_bias'] == 0:
                    net_neuron.setValue(values[neuron_counter])
                    net_neuron.setSum(values[neuron_counter])
            net_layer.setNeuron(neuron_counter + 1, net_neuron)

        net.setLayer(layer_counter, net_layer)
    
    conn.close()
    net.setInputs(values)

    # do the actual prediction
    net.forwardPropagate()
    return net


def single_result_factory(cursor, row):
    return row[0]


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
    if (inputfile is None):
        print('the name of the network must be set!')
        exit(1)
    net = Net(inputfile, int(learning_rate))
    results = predict(net, [0.1001, 0.0002])

    if results.getLayer(len(results.getLayers()) - 1).getValues() > 0.5:
        print("activated ", results.getLayer(len(results.getLayers()) - 1).getValues() )
    else:
        print("not activated ", results.getLayer(len(results.getLayers()) - 1).getValues())
