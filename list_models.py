#!/usr/bin/env python3
import sqlite3
import dotenv
import os

if __name__ == "__main__":
    dotenv.load_dotenv('.env')
    if 'testing' in os.environ and os.environ['testing'] == 1:
        dotenv.load_dotenv('.env.testing')
    else:
        dotenv.load_dotenv('.env')
    
    conn = sqlite3.connect('data/' + os.environ['DB_NAME'])
    c = conn.cursor()
    c.execute('''SELECT * FROM models''')
    print(c.fetchall())
    conn.commit()
    conn.close()
