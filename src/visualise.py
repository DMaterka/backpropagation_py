import matplotlib.pyplot as plt
import numpy as np
from src import backpropagation


def print_learning_curve(curve_data):
    plt.plot(range(0, len(curve_data)), curve_data)
    plt.tight_layout()
    plt.show()


def print_network(net: backpropagation.Net):
    """ Print network structure - neuron and connected weights"""
    fig, axs = plt.subplots()
    posx = 10
    radius = 10
    axs.set_xlim((0, radius*10))
    axs.set_ylim((0, radius*10))
    
    for layer_index in range(len(net.getLayers())):
        interval = 100 / (len(
            net.getLayer(layer_index).getNeurons()) + 1)
        posy = interval
        for neuron_index in range(len(net.getLayer(layer_index).getNeurons())):
            # axs.add_artist(plt.Circle((posx, posy), radius))
            plt.scatter(posx, posy, 1000)
            text_to_show = 'sum:' + '{:.2f}'.format(
                float(net.getLayer(layer_index).getNeuron(neuron_index).getSum()))
            text_to_show += "\n" + 'value:' + "{:.2f}".format(
                float(net.getLayer(layer_index).getNeuron(neuron_index).getValue())
            )
            plt.text(posx, posy, text_to_show, fontsize=12)
            if layer_index > 0:
                weights = ''
                for weight_index in range(0, len(net.getLayer(layer_index).getNeuron(neuron_index).getWeights())):
                    weight_value = net.getLayer(layer_index).getNeuron(neuron_index).getWeights()[weight_index]
                    if np.ndim(weight_value) == 0:
                        weight_value = np.expand_dims(
                            net.getLayer(layer_index).getNeuron(neuron_index).getWeights()[weight_index], 1
                        )
                    weights += '{:.2f}'.format(weight_value[0]) + "\n"
                plt.text(posx - (radius * 2), posy, weights, fontsize=12)
            posy += interval
            net.getLayer(layer_index).getNeuron(neuron_index).setPosition([posx, posy])
        posx += radius * 4
    
    fig.tight_layout()
    plt.show()


def print_decision_regions(training_sets, net: backpropagation.Net):
    inputs = [[], []]
    colours = ['c', 'm', 'y', 'k']
    for inp in range(len(training_sets)):
        init, expected = training_sets[inp]
        for i in range(len(init)):
            inputs[i].append(init[i])
            net.getLayer(0).setNeurons(init)
            net.setExpectedResults(expected)
            net.forwardPropagate()
        colour_index = int(round(net.get_results()[0]))
        assigned_colour = colours[colour_index]
        plt.scatter(init[0], init[1], s=30, c=assigned_colour, label=colour_index)
    plt.legend()
    plt.show()