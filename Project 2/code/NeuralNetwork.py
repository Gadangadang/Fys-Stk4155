import numpy as np
import os
import sys
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


class NeuralNetwork:
    def __init__(self,
                 X,
                 y,
                 num_hidden_layers=10,
                 num_hidden_nodes=60,
                 batch_size=100,
                 eta=0.001,
                 lmbd=0.0,
                 seed=4155,
                 activation="sigmoid"):

        self.X = X  # Design matrix
        self.y = y  # Target
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_nodes = num_hidden_nodes
        self.num_output_nodes = y.shape[1]
        self.batch_size = batch_size
        self.eta = eta
        self.lmbd = lmbd
        self.seed = seed
        self.lmbd = 0
        self.create_layers()

        if activation == "sigmoid":
            self.activation = self.sigmoid_activation
        elif activation == "relu":
            self.activation = self.RELU_activation
        elif activation == "leaky_relu":
            self.activation = self.Leaky_RELU_activation

        self.create_biases_and_weights()

    def create_layers(self):
        self.layers = [np.zeros((self.X.shape[1], self.num_hidden_nodes), dtype=np.float64)
                       for i in range(self.num_hidden_layers)]
        self.layers.insert(0, self.X.copy())
        self.layers.append(
            np.zeros((self.num_output_nodes, self.X.shape[1]), dtype=np.float64))


    def create_biases_and_weights(self):
        np.random.seed(self.seed)
        num_hidden_layers = self.num_hidden_layers
        num_features = self.X.shape[1]
        num_hidden_nodes = self.num_hidden_nodes
        # num_output = self.
        bias_shift = 0.01

        self.weights = [np.random.randn(
            num_hidden_nodes, num_hidden_nodes) for i in range(self.num_hidden_layers - 1)]

        self.weights.insert(0, np.random.randn(
            num_features, num_hidden_nodes))

        self.weights.append(np.random.randn(
            num_hidden_nodes, self.num_output_nodes))

        self.bias = np.ones(num_hidden_layers + 1) * bias_shift

        self.delta_nabla_b = self.bias.copy()
        self.delta_nabla_w = self.weights.copy()

    def update_parameters(self):
        self.backpropagation()

        nabla_b = np.asarray(
            [db + self.lmbd * b for b, db in zip(self.bias, self.delta_nabla_b)])
        nabla_w = np.asarray(
            [dw + self.lmbd * w for w, dw in zip(self.weights, self.delta_nabla_w)])
        self.weights = [w - self.eta * dw for w,
                        dw in zip(self.weights, nabla_w)]
        self.bias = [b - self.eta * db for b, db in zip(self.bias, nabla_b)]

    def feed_forward(self):

        for i in range(self.num_hidden_layers):
            self.layers[i + 1] = self.activation(
                np.matmul(self.layers[i], self.weights[i]) + self.bias[i])

        #-- No activation for last layer --#
        self.layers[-1] = np.matmul(self.layers[-2],
                                    self.weights[-1]) + self.bias[-1]

    def backpropagation(self):
        """
        Returns the derivatives of the cost functions
        """

        error = self.layers[-1] - self.y
        # print(np.sum(error))
        self.delta_nabla_b[-1] = np.sum(error, axis=0)[0]
        self.delta_nabla_w[-1] = np.matmul(self.layers[-2].T, error)
        for i in range(1, self.num_hidden_layers + 1):
            error = np.matmul(
                error, self.weights[-i].T) * self.layers[-i - 1] * (1 - self.layers[-i - 1])
            self.delta_nabla_b[-i - 1] = np.sum(error, axis=0)[0]
            self.delta_nabla_w[-i -
                               1] = np.matmul(self.layers[-i - 2].T, error)

    def run_network(self, epochs):
        for epoch in range(epochs):
            self.feed_forward()
            self.update_parameters()
    def predict(self,x):
        seflf

        self.feed_forward()

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
        text += "Hidden layers:      {} \n".format(self.num_hidden_layer)
        text += "Hidden nodes:       {} \n".format(self.num_hidden_nodes)
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

    beta = OLS_regression(X, z)

    z_ols = X @ beta

    NN = NeuralNetwork(X, z)
    NN.run_network(100)
    z_nn = NN.layers[-1]

    print("Neural Network", MSE(z_nn, z))

    print("OLS", MSE(z_ols, z))


#
