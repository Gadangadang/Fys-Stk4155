import numpy as np
import os
import sys


class NeuralNetwork:
    def __init__(self,
                 X,
                 y,
                 num_hidden_layers=3,
                 num_hidden_nodes=50,
                 batch_size=100,
                 eta=0.1,
                 lmbd=0.0,
                 seed=4155):

        self.X = X  # Design matrix
        self.y = y  # Target
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_nodes = num_hidden_nodes
        self.num_output_nodes = len(y)
        self.batch_size = batch_size
        self.eta = eta
        self.lmbd = lmbd
        self.seed = seed
        self.lmbd = 0

        self.layers = [np.zeros((self.X.shape[1], self.num_hidden_nodes), dtype=np.float64)
                       for i in range(self.num_hidden_layers)]
        self.layers.insert(0, self.X)
        self.layers.append(
            np.zeros((self.num_output_nodes, self.X.shape[1]), dtype=np.float64))

    def create_biases_and_weights(self):
        np.random.seed(self.seed)
        num_hidden_layers = self.num_hidden_layers
        num_features = self.X.shape[1]
        num_hidden_nodes = self.num_hidden_nodes
        # num_output = self.
        bias_shift = 0.01

        self.weights = [np.random.randn(num_hidden_nodes, num_hidden_nodes) for i in range(self.num_hidden_layers - 1)]

        self.weights.insert(0, np.random.randn(
            num_features, num_hidden_nodes))

        self.weights.append(np.random.randn(
            num_hidden_nodes, self.num_output_nodes))

        self.bias = np.ones(num_hidden_layers+1) * bias_shift

        # self.output_weights = np.random.randn(num_hidden_nodes, num_output)
        # self.output_bias = np.zeros(self.n_categories) + 0.01

    def update_parameters(self, batch, eta):
        x,y = batch
        delta_nabla_b, delta_nabla_w = self.backpropagation(x, y)
        nabla_b = np.asarray([db + self.lmbd*b for b, db in zip(self.bias, delta_nabla_b])
        nabla_w = np.asarray([dw + self.lmbd*w for w, dw in  zip(self.weights, delta_nabla_w])
        self.weights = [w-eta*dw for w, dw in zip(self.weights, nabla_w)]
        self.bias = [b-eta*db for b, db in zip(self.bias, nabla_b)]

    def feed_forward(self):

        for i, _ in enumerate(self.layers-1):
            print(self.layers[i+1], self.layers[i], self.weights[i])
            self.layers[i+1] = np.matmul( self.layers[i], self.weights[i] ) + self.bias[i]



    def backpropagation(self):
        """
        Returns the derivatives of the cost functions
        """
        return (delta_nabla_b, delta_nabla_w)

    def sigmoid_activation(self, value):
        return 1 / (1 + np.exp(-value))

    def RELU_activation(self, value):
        if value > 0:
            return value
        else:
            return value

    def Leaky_RELU_activation(self, value):
        if value > 0:
            return value
        else:
            return 0.01 * value

    def __str__(self):
        text = "Information of the Neural Network \n"
        text += "Hidden layers:      {} \n".format(self.hidden_layer.shape[0])
        text += "Hidden nodes:       {} \n".format(self.hidden_layer.shape[1])
        text += "Output nodes:       {} \n".format(self.num_output_nodes)
        text += "Number of features: {} \n".format(self.X.shape[1])

        return text


if __name__ == "__main__":
    # Get modules from project 1
    path = os.getcwd()  # Current working directory
    path += '/../../Project 1/code'
    sys.path.append(path)
    from Functions import *

    #--- Create data from Franke Function ---#
    N = 5               # Number of points in each dimension
    z_noise = 0.2       # Added noise to the z-value
    n = 2               # Highest order of polynomial for X

    x, y, z = generate_data(N, z_noise)
    X = create_X(x, y, n)

    NN = NeuralNetwork(X, z)
    NN.create_biases_and_weights()
    NN.feed_forward()



#
