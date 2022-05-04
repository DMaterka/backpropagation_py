#!/usr/bin/env python3
import dotenv
import os
from src import dbops

if __name__ == "__main__":
    dotenv.load_dotenv('.env')
    if os.environ['TESTING']:
        dotenv.load_dotenv('.env.testing')
        dbpath = 'data/test/' + os.environ['DB_NAME']
    else:
        dotenv.load_dotenv('.env')
        dbpath = 'data/' + os.environ['DB_NAME']

    if os.path.exists(dbpath):
        print('The database already exists')
        exit(1)

    dbops = dbops.DbOps()
    dbops.createSchema(dbpath)
    exit(0)
