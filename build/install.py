#!/usr/bin/env python
import sqlite3
import os
import dotenv


def createSchema(conn):
	
	c = conn.cursor()
	c.execute('''CREATE TABLE models (id integer PRIMARY KEY, name text, error real)''')
	c.execute('''CREATE TABLE layers (id integer PRIMARY KEY, layer_index integer, model_id integer)''')
	c.execute('''CREATE TABLE neurons (id integer PRIMARY KEY, neuron_index integer, layer_id integer)''')
	c.execute('''CREATE TABLE weights (id integer PRIMARY KEY, neuron_from integer, neuron_to integer, weight json)''')
	conn.commit()


dotenv.load_dotenv('.env')
conn = sqlite3.connect('data/' + os.environ['DB_NAME'])
createSchema(conn)
conn.close()
