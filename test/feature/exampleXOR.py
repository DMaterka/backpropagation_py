#!/usr/bin/env python3
import train
from src import visualise
import dotenv

if __name__ == "__main__":
    dotenv.load_dotenv('../../.env.testing')
    inputfile = 'xor.csv'
    iterations = 10000
    learning_rate = 0.5
    structure = "[2]"
    batch_size = 1

    training_sets = train.prepare_training_sets(inputfile)
    net = train.prepare_net(structure, learning_rate, training_sets, inputfile)
    training_sets = [training_sets[0]]
    net.getLayer(0).setBias([.35, .35])
    net.getLayer(1).setWeights([[.15, .2], [.25, .3]]).setBias([.6, .6])
    net.getLayer(2).setWeights([[.4, .45], [.5, .55]])

    trained_net = train.perform_training(net, iterations, training_sets, inputfile, batch_size)
    visualise.print_network(net)
