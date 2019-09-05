#!/usr/bin/env python3
import sqlite3

if __name__ == "__main__":
    conn = sqlite3.connect('data/backprop.db')
    c = conn.cursor()
    c.execute('''SELECT * FROM models''')
    print(c.fetchall())
    conn.commit()
    conn.close()
