import numpy as np
import sys
from import_folders import *
import_all_folders()
from plot_set import *
sys.path.insert(1,"../../../Project 1/code/")
from Functions import *
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from NN_functions import *
from NeuralNetwork import NeuralNetwork


if __name__ == "__main__":
    #--- Create data from Franke Function ---#
    N = 10               # Number of points in each dimension
    z_noise = 0.2       # Added noise to the z-value
    n = 8               # Highest order of polynomial for X
    epochs = 1000
    iterations = 1
    batch_size = int(N * N * 0.8)

    x = np.random.uniform(0, 1, N)
    y = np.random.uniform(0, 1, N)

    X = np.array([x, y]).T

    z = FrankeFunction(x, y) + 0.2 * np.random.randn(N)
    z = z.reshape(N, 1)

    X_train, X_test, Z_train, Z_test = train_test_split(X, z, test_size=0.2)
    Z_train, Z_test = mean_scale_new( Z_train, Z_test)

    num_hidden_layers = 2
    num_hidden_nodes = 30
    batch_size = int(50)
    gamma = 0
    seed = 4155
    n_categories = 1
    epochs = int(100)
    batch_size = int(25)

    activations = ["sigmoid", "relu", "leaky_relu"]
    labels = ["Sigmoid", "Relu", "Leaky relu"]
    linestyles = ["-", "-", "--"]
    cost_func = "MSE"


    plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
    for i, activation in enumerate(activations):
        NN = NeuralNetwork(X,
                           z,
                           num_hidden_layers,
                           num_hidden_nodes,
                           batch_size,
                           0.01,
                           0.0,
                           gamma,
                           seed,
                           activation = activation,
                           cost = cost_func,
                           loss = "MSE",
                           callback = False)
        NN.train_network_stochastic(int(epochs))
        epoch = len(NN.score)
        epochs_lin = np.linspace(0, epoch, epoch)
        plt.plot(epochs_lin,
                 NN.score, label = labels[i], linestyle=linestyles[i],
                 marker = "None", markersize=3)
    plt.xlabel("epoch", fontsize=14)
    plt.ylabel("MSE", fontsize=14)
    plt.title("Activation function comparison for Franke")
    plt.legend(fontsize = 13)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    plt.show()
