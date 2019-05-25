#!/usr/bin/env python
import sqlite3

def createSchema(conn):
	    c = conn.cursor()
	    c.execute('''CREATE TABLE test
	                 (id int, test text)''')
	    c.execute('''INSERT INTO test VALUES(1,'test')''')
	    conn.commit()	

conn = sqlite3.connect('data/backprop.db')
createSchema(conn)
conn.close	 