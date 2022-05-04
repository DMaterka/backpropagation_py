#!/usr/bin/env python3
import sqlite3
import dotenv
import os

if __name__ == "__main__":
    dotenv.load_dotenv('.env')
    if os.environ['TESTING']:
        dotenv.load_dotenv('.env.testing')
        conn = sqlite3.connect('data/test/' + os.environ['DB_NAME'])
    else:
        dotenv.load_dotenv('.env')
        conn = sqlite3.connect('data/' + os.environ['DB_NAME'])

    c = conn.cursor()
    c.execute('''SELECT * FROM models''')
    print(c.fetchall())
    conn.commit()
    conn.close()
