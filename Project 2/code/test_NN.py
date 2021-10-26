import numpy as np
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns

import os
import sys
# Get modules from project 1
path = os.getcwd()  # Current working directory
path += '/../../Project 1/code'
sys.path.append(path)
from Functions import *



from tensorflow import keras
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential      #This allows appending layers to existing models
from tensorflow.keras.layers import Dense           #This allows defining the characteristics of a particular layer
from tensorflow.keras import optimizers             #This allows using whichever optimiser we want (sgd,adam,RMSprop)
from tensorflow.keras import regularizers           #This allows using whichever regularizer we want (l1,l2,l1_l2)


def test_feed_forward():
    # Create test data
    X = np.array(  [[0, 1, 2],
                    [0, 1, 2],
                    [0, 1, 2]])
    y = np.array([1, 2, 3]).reshape(3,1)

    # Set up NN
    num_hidden_layers = 1
    num_hidden_nodes = 2
    NN = NeuralNetwork(  X,
                         y,
                         num_hidden_layers=num_hidden_layers,
                         num_hidden_nodes=num_hidden_nodes,
                         batch_size=100,
                         eta=0.1,
                         lmbd=0.0,
                         seed=4155,
                         activation = "sigmoid")

    # Redefine weights and bias to simple numbers
    W0 = 1
    W1 = 2
    W3 = 3
    NN.weights[0][:,0] = W0
    NN.weights[0][:,1] = W1
    NN.weights[1][:,0] = W3


    # Get computed values after feed forward
    NN.feed_forward()
    layer1_computed = NN.layers[0]
    layer2_computed = NN.layers[1]
    layer3_computed = NN.layers[2]

    # Calculate expected feed forward values
    layer1_expected = X

    layer2_expected = np.zeros((3,2))
    layer2_expected[:,0] = X[:,0]*W0 + X[:,1]*W0 + X[:,2]*W0
    layer2_expected[:,1] = X[:,0]*W1 + X[:,1]*W1 + X[:,2]*W1
    layer2_expected = NN.sigmoid_activation(layer2_expected)

    layer3_expected = layer2_expected[:,0]*W3 + layer2_expected[:,1]*W3

    # Evaluate computed vs expected
    tol = 1e-10
    con1 = (layer1_expected - layer1_computed < tol).all()
    con2 = (layer2_expected - layer2_computed < tol).all()
    con3 = (layer3_expected - layer3_computed < tol).all()
    success = np.array([con1, con2, con3]).all()
    msg = f"Feed forward failed to reproduxe expected results"

    assert success, msg


def test_NN_Franke():
    #--- Create data from Franke Function ---#
    N = 5               # Number of points in each dimension
    z_noise = 0.2       # Added noise to the z-value
    n = 2               # Highest order of polynomial for X

    x, y, z = generate_data(N, z_noise)
    X = create_X(x, y, n)


    # Define settings for Neural Network
    num_hidden_layers = 2
    num_hidden_nodes = 2
    activation='sigmoid'
    eta = 0.01
    lmbd = 0
    epochs = 1000
    batch_size = len(z)
    n_categories = z.shape[1]

    # Our NeuralNetwork class (NN)
    NN = NeuralNetwork(  X,
                         z,
                         num_hidden_layers=1,
                         num_hidden_nodes=num_hidden_nodes,
                         batch_size=batch_size,
                         eta=eta,
                         lmbd=lmbd,
                         seed=4155,
                         activation = activation)
    NN.run_network(epochs)

    # Tensorflow (tf) network
    model = Sequential()
    model.add(Input(shape=X.shape[1]))
    model.add(Dense(num_hidden_nodes, activation=activation, kernel_regularizer=regularizers.l2(lmbd)))
    model.add(Dense(num_hidden_nodes, activation=activation, kernel_regularizer=regularizers.l2(lmbd)))
    model.add(Dense(n_categories))

    sgd = optimizers.SGD(learning_rate=eta,  momentum=0.0)
    model.compile(loss='mean_absolute_error', optimizer=sgd, metrics=['accuracy'])
    model.fit(X, z, epochs=epochs, batch_size=batch_size, verbose=0)


    # Evaluate results
    NN_pred = NN.predict(X) # Our prediction
    tf_pred = model.predict(X) # Tensorflow prediction

    MSE_NN = MSE(z, NN_pred)
    MSE_tf = MSE(z, tf_pred)


    success = MSE_NN < MSE_tf + tol
    msg = f"Neural Network learning is noticeable worse than acchived with tensorflow \n\
            prediction MSE: NN = {MSE_NN:g}, tensorflow = {MSE_tf:g}"

    assert success, msg






def WIP_test_logical_gates():
    # Input for logical gates
    x1 = np.array([0, 0, 1, 1])
    x2 = np.array([0, 1, 0, 1])
    X = np.array([x1, x2]).T # Design matrix

    # Different logical gates
    y_AND = np.array([0, 0, 0, 1]) # And
    y_OR =  np.array([0, 1, 1, 1]) # Or
    y_XOR = np.array([0, 1, 1, 0]) # Exclusive OR

    y_gates = [y_AND, y_OR, y_XOR]

    num_hidden_layers = 2
    num_hidden_nodes = 10
    n_categories = 1

    eta = 1e-2
    lmbd = 0
    epochs = 10000
    batch_size = 4

    for i, y in enumerate(y_gates):
        sklearn_pred, sklearn_accuracy = sklearn_NN(X, y_OR, eta, lmbd, epochs, num_hidden_layers, num_hidden_nodes, n_categories, 'logistic')
        NN = NeuralNetwork(X, y.reshape(4,1), num_hidden_layers, num_hidden_nodes, batch_size, eta, lmbd, seed=4155, activation = "sigmoid", cost = "binary_difference")
        NN.run_network_stochastic(epochs)
        print(np.around(NN.layers_a[-1]))
        accuracy = NN.accuracy_score()
        print(f"{int(accuracy*len(NN.t))} / {len(NN.t)} accurate predictions")

    success = True
    msg = "..."


    assert success, msg


def sklearn_NN(X, y, eta, lmbd, epochs, num_hidden_layers, num_hidden_nodes, n_categories, activation_func):
    n_inputs, n_features = X.shape
    hidden_layer_sizes = [num_hidden_nodes for i in range(num_hidden_layers)]
    dnn = MLPClassifier(hidden_layer_sizes =  hidden_layer_sizes,
                        activation = activation_func,
                        alpha= lmbd,
                        learning_rate_init = eta,
                        max_iter = epochs   )
    dnn.fit(X, y)
    test_pred = dnn.predict(X)
    test_accuracy = accuracy_score(y, test_pred)
    return test_pred, test_accuracy




# def sklearn_network_Morten(X, y):
#
#     # Defining the neural network
#     n_inputs, n_features = X.shape
#     num_hidden_layers = 1
#     num_hidden_nodes = 2
#     hidden_layer_sizes = [num_hidden_nodes for i in range(num_hidden_layers)]
#
#     n_categories = 1
#     activation_func = 'logistic'
#
#
#     # Hyperparameters grid
#     eta_vals = np.logspace(-5, 1, 7)
#     lmbd_vals = np.logspace(-5, 1, 7)
#
#
#     # store models for later use
#     DNN_scikit = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
#     epochs = 1000
#
#     for i, eta in enumerate(eta_vals):
#         for j, lmbd in enumerate(lmbd_vals):
#             dnn = MLPClassifier(hidden_layer_sizes = hidden_layer_sizes,
#                                 activation = activation_func,
#                                 alpha= lmbd,
#                                 learning_rate_init = eta,
#                                 max_iter = epochs   )
#
#             dnn.fit(X, y)
#             DNN_scikit[i][j] = dnn
#             # print("Learning rate  = ", eta)
#             # print("Lambda = ", lmbd)
#             # print("Accuracy score on data set: ", dnn.score(X, y))
#             # print()
#
#
#
#     sns.set()
#     test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
#     for i in range(len(eta_vals)):
#         for j in range(len(lmbd_vals)):
#             dnn = DNN_scikit[i][j]
#
#             test_pred = dnn.predict(X)
#             print(test_pred)
#             test_accuracy[i][j] = accuracy_score(y, test_pred)
#
#     fig, ax = plt.subplots()
#     sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
#     ax.set_title("Test Accuracy")
#     ax.set_ylabel("$\eta$")
#     ax.set_xlabel("$\lambda$")
#     plt.show()




if __name__ == "__main__":
    # test_logical_gates()
    # test_NN_Franke()
    WIP_test_logical_gates()



#
