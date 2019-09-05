#!/usr/bin/env python
import sqlite3


def createSchema(conn):
	
	c = conn.cursor()
	c.execute('''CREATE TABLE models (id integer PRIMARY KEY, name text, error real)''')
	c.execute('''CREATE TABLE layers (id integer PRIMARY KEY, layer_index integer, model_id integer)''')
	c.execute('''CREATE TABLE neurons (id integer PRIMARY KEY, neuron_index integer, layer_id integer)''')
	c.execute('''CREATE TABLE weights (id integer PRIMARY KEY, neuron_from integer, neuron_to integer, weight real)''')
	conn.commit()


conn = sqlite3.connect('data/backprop.db')
createSchema(conn)
conn.close()
