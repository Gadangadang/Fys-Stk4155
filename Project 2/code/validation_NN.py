from NeuralNetwork import NeuralNetwork
from find_hyperparameters import *

import numpy as np
import matplotlib.pyplot as plt


def logic_gates():
    # Input for logic gates
    x1 = np.array([0, 0, 1, 1])
    x2 = np.array([0, 1, 0, 1])
    X = np.array([x1, x2]).T # Design matrix

    # Types of logical gates
    y_AND = np.array([0, 0, 0, 1]) # And
    y_OR =  np.array([0, 1, 1, 1]) # Or
    y_XOR = np.array([0, 1, 1, 0]) # Exclusive OR

    # Output for logic gates
    y_gates = [y_AND, y_OR, y_XOR]


    # NN architecture
    num_hidden_layers = 1
    num_hidden_nodes = 10
    batch_size = 4 # Full GD
    n_categories = 1
    eta = np.logspace(-5, -1, 5)
    lmbds = np.logspace(-5, -1, 5)
    activation = "sigmoid"
    cost_func = "cross_entropy"
    epochs = 10

    for i, y in enumerate(y_gates):
        # sklearn_pred, sklearn_accuracy = sklearn_NN(X, y_OR, eta, lmbd, epochs, num_hidden_layers, num_hidden_nodes, n_categories, 'logistic')
        y = y.reshape(4,1)


        find_hyperparameters(X,
                                 y,
                                 epochs,
                                 num_hidden_layers,
                                 num_hidden_nodes,
                                 batch_size,
                                 etas,
                                 lmbds,
                                 activation,
                                 cost_func,
                                 name = 0,
                                 scaling = "none",
                                 return_best = False)



        # NN = NeuralNetwork(X, y,
        #                          num_hidden_layers,
        #                          num_hidden_nodes,
        #                          batch_size,
        #                          eta=0.001,
        #                          lmbd=0.00,
        #                          gamma = 0.0,
        #                          seed=4155,
        #                          activation="sigmoid",
        #                          cost="MSE")


        # NN.train_network_stochastic(epochs)
        # accuracy = NN.accuracy_score(X, y)
        # print(accuracy)


        exit()





if __name__ == "__main__":
    logic_gates()
