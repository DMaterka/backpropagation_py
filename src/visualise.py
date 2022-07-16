import matplotlib
import matplotlib.pyplot as plt
from src import operations


def print_learning_curve(curve_data):
    matplotlib.use('TkAgg')
    plt.plot(range(0, len(curve_data)), curve_data)
    plt.tight_layout()
    plt.show()


def print_network(net: operations.Net):
    matplotlib.use('TkAgg')
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
            text_to_show += "\n" + 'value:' + "{:.4f}".format(
                float(net.getLayer(layer_index).getNeuron(neuron_index).getValue())
            )
            plt.text(posx, posy, text_to_show, fontsize=12)
            if layer_index > 0:
                weights = ''
                for weight_index in range(0, len(net.getLayer(layer_index).getNeuron(neuron_index).getWeights())):
                    weight_value = net.getLayer(layer_index).getNeuron(neuron_index).getWeights()[weight_index]
                    # Multidimensional weights feature ... to be skipped for now
                    # if np.ndim(weight_value) == 0:
                    #     weight_value = np.expand_dims(weight_value, 1)
                    weights += '{:.2f}'.format(weight_value) + "\n"
                plt.text(posx - (radius * 2), posy, weights, fontsize=12)
            posy += interval
            net.getLayer(layer_index).getNeuron(neuron_index).setPosition([posx, posy])
        posx += radius * 4
    
    fig.tight_layout()
    plt.show()


def print_decision_regions(training_sets, net: operations.Net):
    data = {0: {0: [], 1: []}, 1: {0: [], 1: []}}
    for inp in range(len(training_sets)):
        init, expected = training_sets[inp]
        color_index = operations.Operations().predict(net, init)
        data[color_index][0].append(init[0])
        data[color_index][1].append(init[1])

    colors = ['c', 'm', 'y', 'k']

    for key in range(len(data)):
        assigned_color = colors[key]
        plt.scatter(data[key][0], data[key][1], s=30, c=assigned_color, label=key)

    plt.legend()
    plt.show()
