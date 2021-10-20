import numpy as np
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns




def test_intialize():
    pass

def test_feed_forward():
    pass



def que_test_logical_gates():
    # Input for logical gates
    x1 = np.array([0, 0, 1, 1])
    x2 = np.array([0, 1, 0, 1])
    X = np.array([x1, x2]).T # Design matrix

    # Different logical gates
    y_AND = np.array([0, 0, 0, 1]) # And
    y_OR =  np.array([0, 1, 1, 1]) # Or
    y_XOR = np.array([0, 1, 1, 0]) # Exclusive OR

    y_gates = [y_AND, y_OR, y_XOR]

    num_hidden_layers = 1
    num_hidden_nodes = 2
    n_categories = 2
    activation_func = 'logistic'
    eta = 1e-4
    lmbd = 1e-2
    epochs = 100

    for i, y in enumerate(y_gates):
        test_pred, test_accuracy = sklearn_NN(X, y_OR, eta, lmbd, epochs, num_hidden_layers, num_hidden_nodes, n_categories, activation_func)

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
#     n_categories = 2
#     activation_func = 'logistic'
#
#
#     # Hyperparameters grid
#     eta_vals = np.logspace(-5, 1, 7)
#     lmbd_vals = np.logspace(-5, 1, 7)
#
#     # store models for later use
#     DNN_scikit = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
#     epochs = 100
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
#             test_accuracy[i][j] = accuracy_score(y, test_pred)
#
#     fig, ax = plt.subplots()
#     sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
#     ax.set_title("Test Accuracy")
#     ax.set_ylabel("$\eta$")
#     ax.set_xlabel("$\lambda$")
#     plt.show()



if __name__ == "__main__":
    # sklearn_network_Morten(X, y_OR)
    pass




#
