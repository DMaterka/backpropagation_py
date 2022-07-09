import numpy as np


class ActivationFn:
    @staticmethod
    def sigmoid(x):
        x = np.array(x)
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoidprime(x):
        return (__class__.sigmoid(x)) * (1 - __class__.sigmoid(x))

    @staticmethod
    def tanh(z):
        # todo
        return np.tanh(z)

    @staticmethod
    def relu(z):
        if z > 0:
            return input
        return 0

    @staticmethod
    def relu_prime(z):
        if z > 0:
            return 1
        return 0
