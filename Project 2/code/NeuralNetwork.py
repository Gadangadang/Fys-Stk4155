import autograd.numpy as np
import os
import sys
from autograd import elementwise_grad as grad
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
                 activation="sigmoid",
                 cost = "difference"):

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
        self.create_biases_and_weights()

        if activation == "sigmoid":
            self.activation = self.sigmoid_activation
            self.activation_der = grad(self.sigmoid_activation)
        elif activation == "relu":
            self.activation = self.RELU_activation
        elif activation == "leaky_relu":
            self.activation = self.Leaky_RELU_activation
        elif activation == "soft_max":
            self.activation = self.soft_max_activation
        if cost == "difference":
            self.cost = self.difference
            self.cost_der = grad(self.difference)
        elif cost == "binary_difference":
            self.cost = self.binary_difference
            self.cost_der = grad(self.binary_difference)


    def create_layers(self):
        self.layers = [np.zeros((self.X.shape[1], self.num_hidden_nodes), dtype=np.float64)
                       for i in range(self.num_hidden_layers)]
        self.layers.insert(0, self.X.copy())
        self.layers.append(
            np.zeros((self.num_output_nodes, self.X.shape[1]), dtype=np.float64))
        self.layers = np.asarray(self.layers)
        self.layers_UA = np.copy(self.layers)

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
        self.weights = np.asarray(self.weights)
        self.delta_nabla_b = self.bias.copy()
        self.delta_nabla_w = self.weights.copy()

    def update_parameters(self):
        self.backpropagation()
        nabla_b = np.asarray([db + self.lmbd * b for b, db in zip(self.bias, self.delta_nabla_b)])
        nabla_w = np.asarray([dw + self.lmbd * w for w, dw in zip(self.weights, self.delta_nabla_w)])

        self.weights = np.asarray([w - self.eta * dw for w,
                        dw in zip(self.weights, nabla_w)])
        self.bias = np.asarray([b - self.eta * db for b, db in zip(self.bias, nabla_b)])

    def feed_forward(self):

        for i in range(self.num_hidden_layers):
            val = np.dot(self.layers[i], self.weights[i]) + self.bias[i]
            self.layers[i + 1] = self.activation(val)
            self.layers_UA[i+1] = val
        #-- No activation for last layer --#
        self.layers[-1] = np.matmul(self.layers[-2],
                                    self.weights[-1]) + self.bias[-1]

    def backpropagation(self):
        """
        Returns the derivatives of the cost functions
        """
        print(self.layers_UA[-1])
        print(self.activation_der(np.zeros(10)))
        exit()
        delta = self.cost_der(self.layers[-1])*self.activation_der(self.layers_UA[-1])

        self.delta_nabla_b[-1] = delta
        self.delta_nabla_w[-1] = np.matmul(delta, self.layers[-2].T)
        for i in range(2, self.num_hidden_layers ):
            delta = np.matmul(delta, self.weights[-i+1].T)
            self.delta_nabla_b[-i - 1] = delta
            self.delta_nabla_w[-i - 1] = np.matmul(delta, self.layers[-i-1].T)

    def predict(self, X = None):
        if X != None:
            self.layers[0] = X
        self.feed_forward()
        return self.layers[-1]

    def run_network(self, epochs):
        for epoch in range(epochs):
            self.feed_forward()
            self.update_parameters()


    """
    Activation funtions
    """
    def sigmoid_activation(self, value):
        return 1 / (1 + np.exp(-value))
    def sigmoid_activation_man_der(self, value):
        sig = self.sigmoid_activation(value)
        return sig*(sig-1)

    def RELU_activation(self, value):
        vals = np.where(value > 0, value, 0)
        return vals

    def Leaky_RELU_activation(self, value):
        vals = np.where(value > 0, value, 0.01 * value)
        return vals
    def soft_max_activation(self, value):
        val_exp = np.exp(value)
        return val_exp/(np.sum(val_exp))

    def indicator(self):
        val = np.sum(self.layers[-1] == self.y)/len(self.y)
        return val


    """
    Cost funtions
    """
    def difference(self, y_tilde):
        return  y_tilde - self.y

    def binary_difference(self, weights, bias):
        y_pred = self.predict( weights, bias)
        return -(self.y*np.log(y_pred)+ (1-self.y)*np.log(1-y_pred))


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
    NN.run_network(int(100))

    print("Neural Network", MSE(z, NN.predict( X)))

    print("     OLS      ", MSE(z_ols, z))

    """a = np.zeros(10, dtype=np.float64)

    print(np.exp(a))"""


#
