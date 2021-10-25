import os
import sys
import autograd.numpy as np
from autograd import elementwise_grad
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


class NeuralNetwork:
    def __init__(self,
                 X,
                 t,
                 num_hidden_layers=2,
                 num_hidden_nodes=10,
                 batch_size=100,
                 eta=0.001,
                 lmbd=0.0,
                 seed=4155,
                 activation="sigmoid",
                 cost = "MSE"):

        self.N = X.shape[0]
        self.num_features = X.shape[1]
        self.num_categories = t.shape[1]

        self.X = X  # Design matrix shape: N x features --> features x N
        self.t = t  # Target, shape: categories x N
        # Make clever function to check shapes please (Saki)

        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_nodes = num_hidden_nodes
        self.num_output_nodes = self.num_categories
        self.L = self.num_hidden_layers + 2 # number of layer in total

        self.batch_size = batch_size
        self.eta = eta
        self.lmbd = lmbd
        self.seed = seed
        self.lmbd = 0
        self.create_layers()
        self.create_biases_and_weights()

        if activation == "sigmoid":
            self.activation = self.sigmoid_activation
            self.activation_der = elementwise_grad(self.sigmoid_activation)
        elif activation == "relu":
            self.activation = self.RELU_activation
        elif activation == "leaky_relu":
            self.activation = self.Leaky_RELU_activation
        elif activation == "soft_max":
            self.activation = self.soft_max_activation
        if cost == "MSE":
            self.cost = self.MSE
            self.cost_der = elementwise_grad(self.MSE)
        elif cost == "binary_difference":
            self.cost = self.binary_difference
            self.cost_der = elementwise_grad(self.binary_difference)


    def create_layers(self):
        """
        layers_a: contain activation values of all nodes
        layers_z: contain weighted sum z / unactivated values of all nodes
        """
        self.layers_a = [np.zeros((self.N, self.num_hidden_nodes), dtype=np.float64)
                       for i in range(self.num_hidden_layers)] # Intialized with the hidden layers

        self.layers_a.insert(0, self.X.copy()) # Add input layer
        self.layers_a.append(
        np.zeros((np.shape(self.t)), dtype=np.float64)) # Add output layer
        self.layers_z = self.layers_a.copy()





    def create_biases_and_weights(self):
        np.random.seed(self.seed)
        num_hidden_layers = self.num_hidden_layers
        num_features = self.num_features
        num_hidden_nodes = self.num_hidden_nodes
        num_categories = self.num_categories
        bias_shift = 0.01

        self.weights = [np.random.randn(num_hidden_nodes, num_hidden_nodes) for i in range(self.num_hidden_layers - 1)]
        # self.weights.insert(0, np.random.randn(num_features, num_hidden_nodes))
        self.weights.insert(0, np.random.randn(num_hidden_nodes, num_features))

        self.weights.insert(0, np.nan) # insert unused weight to get nice indexes
        # self.weights.append(np.random.randn(num_hidden_nodes, self.num_output_nodes))
        self.weights.append(np.random.randn(self.num_output_nodes, num_hidden_nodes))


        # Add individual biases?
        self.bias = [np.ones(num_hidden_nodes) for i in range(self.num_hidden_layers)]
        self.bias.insert(0, np.nan) # insert unused bias to get nice indexes
        self.bias.append(np.ones(num_categories))

        # Asarray?!?! (Willy check)
        self.bias = np.asarray(self.bias)
        self.weights = np.asarray(self.weights)

        # self.delta_nabla_w = self.weights.copy()
        # self.delta_nabla_b = self.bias.copy()
        self.local_gradient = self.layers_a.copy() #also called error


    def update_parameters(self):
        self.backpropagation()
        nabla_b = np.asarray([db + self.lmbd * b for b, db in zip(self.bias, self.delta_nabla_b)])
        nabla_w = np.asarray([dw + self.lmbd * w for w, dw in zip(self.weights, self.delta_nabla_w)])

        self.weights = np.asarray([w - self.eta * dw for w,
                        dw in zip(self.weights, nabla_w)])
        self.bias = np.asarray([b - self.eta * db for b, db in zip(self.bias, nabla_b)])

    def feed_forward(self):
        for l in range(1, self.L):
            Z_l = self.layers_a[l-1] @self.weights[l].T # + self.bias[l][np.newaxis, :]
            self.layers_z[l] = Z_l
            self.layers_a[l] = self.activation(Z_l)

    def backpropagation(self):
        """
        Returns the gradient of the cost function
        """
        print("Backpropagation")

        self.local_gradient[-1] = self.cost_der(self.layers_a[-1])*self.activation_der(self.layers_z[-1])



        # self.delta_nabla_b[-1] = delta
        # self.delta_nabla_w[-1] = np.matmul(delta, self.layers_a[-2].T)
        for l in reversed(range(self.L-1)):
            print(np.shape(self.weights[l+1].T))
            print(np.shape(self.local_gradient[l+1]))
            print(np.shape(self.local_gradient[l+1]))
            print(np.shape(self.activation_der(self.layers_z[l])))
            exit()


            self.local_gradient[l] = self.weights[l+1].T @ self.local_gradient[l+1] * self.activation_der(self.layers_z[l])
            exit()

        # for i in range(2, self.num_hidden_layers ):
        #     delta = np.matmul(delta, self.weights[-i+1].T)
        #     self.delta_nabla_b[-i - 1] = delta
        #     self.delta_nabla_w[-i - 1] = np.matmul(delta, self.layers_a[-i-1].T)

    def predict(self, X = None):
        if X != None:
            self.layers_a[0] = X
        self.feed_forward()
        return self.layers_a[-1]

    def run_network(self, epochs):
        for epoch in range(epochs):
            self.feed_forward()
            self.update_parameters()


    """
    Activation funtions
    """
    def sigmoid_activation(self, value):
        return 1.0 / (1.0 + np.exp(-value))
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
        val = np.sum(self.layers_a[-1] == self.y)/len(self.y)
        return val


    """
    Cost funtions
    """


    def MSE(self, y_tilde):
        return (y_tilde - self.t)**2

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
    # The above imports numpy as np so we have to redefine:
    import autograd.numpy as np


    #--- Create data from Franke Function ---#
    N = 5               # Number of points in each dimension
    z_noise = 0.2       # Added noise to the z-value
    n = 2               # Highest order of polynomial for X

    x, y, z = generate_data(N, z_noise)
    X = create_X(x, y, n)

    beta = OLS_regression(X, z)
    z_ols = X @ beta


    NN = NeuralNetwork(X, z)
    NN.feed_forward()
    NN.backpropagation()

    # print("Neural Network", MSE(z, NN.predict( X)))
    #
    # print("     OLS      ", MSE(z_ols, z))




#
