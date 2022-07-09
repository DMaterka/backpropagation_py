import numpy as np
from .DTO.net import Net
from .activation import ActivationFn

debug = 0


class Operations:

    def forwardPropagate(self, net: Net) -> Net:
        """ calculate network values from weights and activation function"""
        for i in range(0, len(net.getLayers()) - 1):
            currentLayer = net.getLayer(i)
            nextLayer = net.getLayer(i + 1)

            """ produce neurons' sums and values """
            for j in range(len(nextLayer.getNeurons())):
                if nextLayer.getNeuron(j).is_bias:
                    continue
                weights = nextLayer.getNeuron(j).getWeights()

                if i == 0:
                    values = currentLayer.getSums()
                else:
                    values = currentLayer.getValues()
                sum = np.dot(weights, values)
                nextLayer.getNeuron(j).setSum(sum)
                nextLayer.getNeuron(j).setValue(ActivationFn().sigmoid(sum))
        return net

    def calculateTotalError(self, net: Net):
        total_error = 0
        for index in range(len(net.getExpectedResults())):
            total_error += np.sum((0.5 * (net.getExpectedResults()[index] - self.get_results(net)[index]) ** 2))
        net.learning_curve_data.append(total_error)
        return total_error

    def backPropagate(self, net: Net):
        self.forwardPropagate(net)
        total_error = self.calculateTotalError(net)
        print(total_error)
        for j in range(len(net.getLayers()) - 1, 0, -1):
            for weight_index in range(len(net.getLayer(j).getWeights())):
                if j == len(net.getLayers()) - 1:
                    # this represents a value of partial derivative of results error dExp_results/dValues
                    partial_error = net.getLayer(j).getValues() - net.getExpectedResults().T
                    # deltaSum is partial derivative of total error with respect to given weight which consists of:
                    # partial derivative of next's neuron value with respect to the sum
                    # times partial error
                    # times next layer partial derivative of sum with respect to a weight
                    deltaSum = ActivationFn().sigmoidprime(net.getLayer(j).getNeuron(weight_index).getSum()) \
                               * partial_error[weight_index] \
                               * net.getLayer(j-1).getValues()
                else:
                    partial_sum = 0
                    for up_neur in range(len(net.getLayer(j+1).getNeurons())):
                        weight = net.getLayer(j + 1).getNeuron(up_neur).getWeights()[weight_index]
                        err_times_upper_delta = (
                                net.getLayer(j + 1).getNeuron(up_neur).getDeltaSum()[up_neur] /
                                net.getLayer(j).getNeuron(up_neur).getValue()
                        )
                        partial_sum += err_times_upper_delta * weight
                    
                    d_val = ActivationFn().sigmoidprime(net.getLayer(j).getNeuron(weight_index).getSum())
                    deltaSum = partial_sum * d_val * net.getLayer(j-1).getSums()

                net.getLayer(j).getNeuron(weight_index).setDeltaSum(deltaSum)

        self.update_weights(net)
        
        # if self.getLayer(j).getBias():
            # bias = self.getLayer(j-1).getBias()
            # biasValue = bias.weights[ds] + (
            #             self.learning_rate * self.getLayer(j).getNeuron(ds).getDeltaSum() *
            #             self.getLayer(j).getNeuron(ds).getValue()
            # )
            # np.append(bias.weights, biasValue)

    def update_weights(self, net):
        for j in range(len(net.getLayers()) - 1, 0, -1):
            for ds in range(len(net.getLayer(j).getWeights())):
                new_weight = net.getLayer(j).getNeuron(ds).getWeights() - \
                             (net.learning_rate * net.getLayer(j).getNeuron(ds).getDeltaSum())
                net.getLayer(j).getNeuron(ds).setWeights(new_weight)
                if not np.array_equal(net.getLayer(j).getNeuron(ds).getWeights(), new_weight):
                    raise Exception("Weights were not saved")

    def get_results(self, net):
        return net.getLayer(len(net.getLayers()) - 1).getValues()
    
