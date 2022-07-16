#!/usr/bin/env python3
import ast
import sys
import getopt
import dotenv
from src.operations import Operations
import src.dbops


if __name__ == "__main__":
    inputfile = ''
    values = ''
    argv = sys.argv[1:]
    learning_rate = 1
    dotenv.load_dotenv('.env')

    try:
        opts, args = getopt.getopt(argv, "hi:v:", ["ifile=", "values="])
    except getopt.GetoptError:
        print('predict.py -i <inputfile> -v <values>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('predict.py -i <inputfile> -v <values>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-v", "--values"):
            values = arg
    if (inputfile is None):
        print('the name of the model must be set!')
        exit(1)

    net = src.dbops.DbOps(inputfile).load_net(inputfile)
    evalueatedVal = ast.literal_eval(values)
    result = Operations().predict(net, evalueatedVal)

    if result == 1:
        print("Activated")
    else:
        print("Not activated")
