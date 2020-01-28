from src import backpropagation
import sqlite3
import json
import os
import ast


class DbOps:
    
    def __init__(self, dbName=''):
        if dbName == '':
            """ A default database name given"""
            dbName = os.environ['DB_NAME']
        if 'TESTING' in os.environ:
            self.db_path = os.environ['PROJECT_ROOT'] + 'test/data/' + dbName
        else:
            self.db_path = 'data/' + dbName
            
        if not os.path.isfile(self.db_path):
            """ If the database doesn't exist, create its structure"""
            self.createSchema(self.db_path)
            
        self.conn = sqlite3.connect(self.db_path)
        
    def createSchema(self, name):
        self.conn = sqlite3.connect(name)
        c = self.conn.cursor()
        c.execute('''CREATE TABLE models (id integer PRIMARY KEY, name text, error real)''')
        c.execute('''CREATE TABLE layers (id integer PRIMARY KEY, layer_index integer, model_id integer)''')
        c.execute(
            '''CREATE TABLE
                neurons (
                    id integer PRIMARY KEY,
                    neuron_index integer,
                    layer_id integer,
                    sum json,
                    value json,
                    is_bias bool DEFAULT FALSE
                )
            '''
        )
        c.execute('''CREATE TABLE weights (id integer PRIMARY KEY, neuron_from integer, neuron_to integer, weight json)''')
        self.conn.commit()
        self.conn.close()
    
    def save_net(self, net: backpropagation.Net, total_error, model_name):
        self.conn.row_factory = sqlite3.Row
        c = self.conn.cursor()
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
            
            bias = net.getLayer(layer_index).getBias()
            if bias:
                sums = []
                c.execute(
                    'INSERT INTO neurons (neuron_index, layer_id, sum, value, is_bias) VALUES (?, ?, ?, ?, ?)',
                    (
                        0, layerid, json.dumps(sums), json.dumps(sums), True
                    )
                )
                bias_id = c.lastrowid
                for prev_layer_neuron_index in range(0, len(net.getLayer(layer_index).getNeurons())):
                    c.execute('SELECT * FROM neurons WHERE neuron_index=? AND layer_id=?',
                              (prev_layer_neuron_index, layerid))
                    current_neuron = c.fetchone()
                    c.execute(
                        'INSERT INTO weights (neuron_from, neuron_to, weight) VALUES (?, ?, ?)',
                        (bias_id, current_neuron['id'], json.dumps(bias.getWeights().tolist()))
                    )
        self.conn.commit()
        self.conn.close()
    
    def update_net(self, net: backpropagation.Net, total_error, model_name):
        self.conn.row_factory = sqlite3.Row
        c = self.conn.cursor()
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
        self.conn.commit()
        self.conn.close()
    
    def load_net(self, model_name):
        self.conn.row_factory = sqlite3.Row
        c = self.conn.cursor()
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
            bias_weights = []
            for neuron in neurons:
                if neuron['is_bias'] == 1:
                    current_bias = backpropagation.Neuron(True)
                    current_bias.setValue(1)
                    c.execute(
                        'SELECT weight FROM weights WHERE neuron_from=?', (neuron['id'],)
                    )
                    db_weights = c.fetchone()['weight']
                    # current_bias.setWeights(db_weights)
                    bias_weights.append(ast.literal_eval(db_weights))
                else:
                    current_neuron = backpropagation.Neuron()
                    current_neuron.setValue(neuron['value'])
                    c.execute(
                        'SELECT weight FROM weights WHERE neuron_to=?', (neuron['id'],)
                    )
                    db_weights = c.fetchone()['weight']
                    current_neuron.setWeights(db_weights)
                    neuron_weights.append(db_weights)
                    current_layer.setNeuron(neuron['neuron_index'], current_neuron)
                
            current_layer.setWeights(neuron_weights)
            if bias_weights:
                # drop bias container dimension in order to keep dimensions
                current_layer.setBias(bias_weights[0])
            net.setLayer(layer['layer_index'], current_layer)
        return net
    
    def delete_model(self, model_name):
        self.conn.row_factory = sqlite3.Row
        c = self.conn.cursor()
        c.execute('SELECT * FROM models WHERE name=?', (model_name,))
        results = c.fetchone()
        modelid = results['id']
        c.execute('SELECT * FROM layers WHERE model_id=?', (modelid,))
        layers = c.fetchall()
        for layer in layers:
            c.execute('SELECT * FROM neurons WHERE layer_id=?', (layer['id'],))
            neurons = c.fetchall()
            for neuron in neurons:
                c.execute(
                    'DELETE FROM weights WHERE neuron_to=?', (neuron['id'],)
                )
            c.execute('DELETE FROM neurons WHERE layer_id=?', (layer['id'],))
        c.execute('DELETE FROM layers WHERE model_id=?', (modelid,))
        c.execute('DELETE FROM models WHERE id=?', (modelid,))
        self.conn.commit()
        self.conn.close()
    
    def get_model_results(self, name):
        if not os.path.exists(self.db_path):
            return 0

        self.conn.row_factory = sqlite3.Row
        c = self.conn.cursor()
        try:
            c.execute('SELECT * FROM models WHERE name=?', (name,))
            results = c.fetchone()
        except:
            return None
            
        return results
