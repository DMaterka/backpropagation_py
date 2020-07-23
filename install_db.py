#!/usr/bin/env python3
import dotenv
import os
from src import dbops

if __name__ == "__main__":
    dotenv.load_dotenv('.env')
    if 'TESTING' in os.environ and os.environ['testing'] == 1:
        dotenv.load_dotenv('.env.testing')
    else:
        dotenv.load_dotenv('.env')

    if os.path.exists('data/' + os.environ['DB_NAME']):
        print('The database already exists')
        exit(1)

    dbops = dbops.DbOps()

    dbops.createSchema('data/' + os.environ['DB_NAME'])
