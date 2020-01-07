from src import backpropagation
import sqlite3
import json
import os
import ast
#TODO make a class with singleton to store connection object
#


def createSchema(name):
    conn = sqlite3.connect('data/' + name)
    c = conn.cursor()
    c.execute('''CREATE TABLE models (id integer PRIMARY KEY, name text, error real)''')
    c.execute('''CREATE TABLE layers (id integer PRIMARY KEY, layer_index integer, model_id integer)''')
    c.execute(
        '''CREATE TABLE neurons (id integer PRIMARY KEY, neuron_index integer, layer_id integer, sum json, value json)'''
    )
    c.execute('''CREATE TABLE weights (id integer PRIMARY KEY, neuron_from integer, neuron_to integer, weight json)''')
    conn.commit()
    conn.close()


def save_net(net: backpropagation.Net, total_error, model_name):
    conn = sqlite3.connect('data/' + os.environ['DB_NAME'])
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('INSERT INTO models (name, error) VALUES (?, ?)', (model_name, total_error))
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
    conn.commit()
    conn.close()


def update_net(net: backpropagation.Net, total_error, model_name):
    conn = sqlite3.connect('data/' + os.environ['DB_NAME'])
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM models WHERE name=?', (model_name,))
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
    conn.commit()
    conn.close()


def load_net(model_name):
    conn = sqlite3.connect('data/' + os.environ['DB_NAME'])
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM models WHERE name=?', (model_name,))
    results = c.fetchone()
    net = backpropagation.Net(results['name'], 0.5)
    modelid = results['id']
    c.execute('SELECT * FROM layers WHERE model_id=?', (modelid,))
    layers = c.fetchall()
    for layer in layers:
        current_layer = backpropagation.Layer()
        c.execute('SELECT * FROM neurons WHERE layer_id=?', (layer['id'],))
        neurons = c.fetchall()
        neuron_weights = []
        for neuron in neurons:
            current_neuron = backpropagation.Neuron()
            evaluated_value = ast.literal_eval(neuron['value'])[0]
            current_neuron.setValue(evaluated_value)
            if layer['layer_index'] > 0:
                c.execute(
                    'SELECT weight FROM weights WHERE neuron_to=?', (neuron['id'],)
                )
                neuron_weights.append(c.fetchall())
                
            current_layer.setNeuron(neuron['neuron_index'], current_neuron)
        current_layer.setWeights(neuron_weights)
        net.setLayer(layer['layer_index'], current_layer)
    return net

def get_model_results(name):
    conn = sqlite3.connect('data/' + os.environ['DB_NAME'])
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM models WHERE name=?', (name,))
    results = c.fetchone()
    return results
