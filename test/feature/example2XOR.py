#!/usr/bin/env python3
import train
from src import visualise
import dotenv

if __name__ == "__main__":
    dotenv.load_dotenv('../../.env.testing')
    inputfile = 'xor2.csv'
    iterations = 10000
    learning_rate = 0.5
    structure = "[3]"
    batch_size = 1

    training_sets = train.prepare_training_sets(inputfile)
    net = train.prepare_net(structure, learning_rate, training_sets, inputfile)

    net.getLayer(1).setWeights([[.3, .5], [.4, .9], [.8, .2]])
    net.getLayer(2).setWeights([[.3, .5, .9]])
    net.setExpectedResults([[0]])

    trained_net = train.perform_training(net, iterations, training_sets, inputfile, batch_size)
    
    visualise.print_network(trained_net)
