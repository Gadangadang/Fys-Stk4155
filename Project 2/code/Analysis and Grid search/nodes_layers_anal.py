from import_folders import *
import_all_folders()
from NeuralNetwork import NeuralNetwork
from NN_functions import *
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import numpy as np



if __name__ == "__main__":
    """Load breast cancer dataset"""

    np.random.seed(0)  # create same seed for random number every time
    cancer = load_breast_cancer()  # Download breast cancer dataset

    # Feature matrix of 569 rows (samples) and 30 columns (parameters)
    inputs = cancer.data
    # Label array of 569 rows (0 for benign and 1 for malignant)
    outputs = cancer.target
    labels = cancer.feature_names[0:30]

    x = inputs
    y = outputs.reshape((len(outputs), 1))

    # Sampling only certain features.
    X = np.reshape(x[:, 1], (len(x[:, 1]), 1))
    features = range(1, 30)
    for i in features:
        temp = np.reshape(x[:, i], (len(x[:, i]), 1))
        X = np.hstack((X, temp))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_test = standard_scale(X_train, X_test)

    lmbd = 0
    eta = 0.01
    batch_size = int(50)
    gamma = 0
    seed = 4155
    n_categories = 1
    epochs = int(200)
    batch_size = int(25)

    num_hidden_nodes = [5, 10, 20, 30, 40]
    num_hidden_layers = [1, 2, 3, 4, 5]

    activation = "sigmoid"
    cost_func = "cross_entropy"
    test_scores = np.zeros((len(num_hidden_nodes), len(num_hidden_layers)))

    for i, nodes in enumerate(num_hidden_nodes):
        for j, layers in enumerate(num_hidden_layers):
            print(f"\r(nodes, layers) = {nodes},{layers}", end="")
            NN = NeuralNetwork(X_train,
                               y_train,
                               layers,
                               nodes,
                               batch_size,
                               eta,
                               lmbd,
                               gamma,
                               seed,
                               activation,
                               cost="cross_entropy",
                               loss="accuracy",
                               callback=True)
            NN.train_network_stochastic(int(epochs))
            test_scores[i][j] = NN.get_score(X_test, y_test)

    plot_heatmap(test_scores, [r"Hidden nodes per layers", num_hidden_nodes], [
                 "Hidden layers", num_hidden_layers], title="Neural network test accuracy for Breast cancer data", name=None)
    indx = np.where(test_scores == np.max(test_scores))
