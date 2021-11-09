import numpy as np
import matplotlib.pyplot as plt

from NeuralNetwork import NeuralNetwork
from find_hyperparameters import *

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score



def sklearn_NN(X, y, eta, lmbd, epochs, num_hidden_layers, num_hidden_nodes, n_categories, activation_func):
    n_inputs, n_features = X.shape
    hidden_layer_sizes = [num_hidden_nodes for i in range(num_hidden_layers)]
    dnn = MLPClassifier(hidden_layer_sizes =  hidden_layer_sizes,
                        activation = activation_func,
                        alpha= lmbd,
                        learning_rate_init = eta,
                        max_iter = epochs )
    dnn.fit(X, y)
    test_pred = dnn.predict(X)
    test_accuracy = accuracy_score(y, test_pred)
    return test_pred, test_accuracy

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
    y_gates = [y_XOR]


    # NN architecture
    num_hidden_layers = 1
    num_hidden_nodes = 4
    batch_size = 2 # Full GD
    n_categories = 1
    eta = 5e-2
    lmbd = 0
    gamma = 0


    etas = np.logspace(-4, -2, 3)
    lmbds = np.logspace(-5, -1, 5)

    activation = "sigmoid"
    cost_func = "cross_entropy"
    cost_func = "MSE"
    epochs = 20

    y = y_XOR.reshape(4,1)
    find_hyperparameters(X, X, y, y,
                             epochs,
                             num_hidden_layers,
                             num_hidden_nodes,
                             batch_size,
                             etas,
                             lmbds,
                             activation,
                             cost_func,
                             name = 0,
                             return_best = False)
    exit()
    for i, y in enumerate(y_gates):
        # sklearn_pred, sklearn_accuracy = sklearn_NN(X, y_OR, eta, lmbd, epochs, num_hidden_layers, num_hidden_nodes, n_categories, 'logistic')
        y = y.reshape(4,1)
        input()


        # find_hyperparameters(X,
        #                          y,
        #                          epochs,
        #                          num_hidden_layers,
        #                          num_hidden_nodes,
        #                          batch_size,
        #                          etas,
        #                          lmbds,
        #                          activation,
        #                          cost_func,
        #                          name = 0,
        #                          scaling = "none",
        #                          return_best = False)


        sklearn_pred, sklearn_accuracy = sklearn_NN(X, y.ravel(), eta, lmbd, epochs, num_hidden_layers, num_hidden_nodes, n_categories, 'logistic')
        NN = NeuralNetwork(X, y,
                                 num_hidden_layers,
                                 num_hidden_nodes,
                                 batch_size,
                                 eta,
                                 lmbd,
                                 gamma,
                                 seed=4155,
                                 activation=activation,
                                 cost=cost_func)


        NN.train_network_stochastic(epochs)
        accuracy = NN.accuracy_score(X, y)
        NN_pred = np.round(NN.predict(X)).ravel()
        NN_pred = NN.predict(X)


        print("sklearn, acc:", sklearn_accuracy)
        print("sklearn, pred:", sklearn_pred)

        print("NN, acc:", accuracy)
        print("NN, pred:", NN_pred)









if __name__ == "__main__":
    logic_gates()
