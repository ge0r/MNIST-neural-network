import numpy as np
import time

class Network(object):
    def __init__(self, sizes):
        self.sizes = sizes
        self.biases = []
        self.activation = []
        self.weights = []

        self.Z = []

        # output has as many rows as the last (-1) layer
        self.output = np.zeros((sizes[-1], 1))

        self.init_layers()
        self.create_biases()
        self.create_weights()

    # loads weights and biases from file network.prm in this module's directory
    def load_params(self):
        pass

    # saves the weights and biases
    def save_params(self):
        pass

    def init_layers(self):
        for neuron_num in self.sizes:
            self.activation.append(np.zeros((neuron_num, 1)))
            self.Z.append(np.zeros((neuron_num, 1)))

    def create_biases(self):
        for neurons in self.sizes:
            self.biases.append(np.random.normal(0, 1, (neurons, 1)))

    def create_weights(self):
        for layer in range(1, len(self.activation)):
            weight = np.random.normal(0, 1, [self.sizes[layer], self.sizes[layer - 1]])
            self.weights.append(weight)

    def set_layer_biases(self, layer, db):
        self.biases[layer] -= db

    def set_layer_weights(self, layer, dw):
        self.weights[layer] -= dw

    def sigmoid(self, z):
        sigma = 1 / (1 + np.exp(-z))
        return sigma

    def activation_diff(self, z):
        sigmoid_prime = np.power(1 + np.exp(-z), -2) * np.exp(-z)
        return sigmoid_prime

    def initialize_input_layer(self, image):
        self.activation[0] = image / 255.

    def feed_forward(self):
        for l in range(1, len(self.activation)):
            # note the difference between expression below and the one on the paper concerning the weight index
            self.Z[l] = np.add(np.dot(self.weights[l-1], self.activation[l-1]), self.biases[l])
            self.activation[l] = self.sigmoid(self.Z[l])

        self.output = self.activation[-1]
        return self.output


