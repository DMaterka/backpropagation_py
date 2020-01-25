#!/usr/bin/env python3
import os

import train
from src import visualise, dbops
import dotenv

if __name__ == "__main__":
    dotenv.load_dotenv('.env.testing')
    inputfile = 'xor.csv'
    outputfile = ''
    iterations = 10000
    learning_rate = 0.5
    structure = "[2]"
    batch_size = 1

    training_sets = train.prepare_training_sets(inputfile)
    net = train.prepare_net(structure, learning_rate, training_sets, inputfile)

    inputLayer = net.getLayer(0)
    inputLayer = inputLayer.setNeurons([[0], [0]])
    inputLayer.setBias([.35, .35])

    hiddenLayer = net.getLayer(1)
    hiddenLayer.setNeurons([[0], [0]])
    hiddenLayer.setWeights([[.15, .2], [.25, .3]])
    hiddenLayer.setBias([.6, .6])
    
    outputLayer = net.getLayer(2)
    outputLayer.setNeurons([[0], [0]])
    outputLayer.setWeights([[.4, .45], [.5, .55]])

    if not os.path.isfile('test/data/' + os.environ['DB_NAME']):
        dbops.createSchema(os.environ['DB_NAME'])
    
    trained_net = train.perform_training(net, iterations, training_sets, inputfile, batch_size)
    
    visualise.print_network(trained_net)
