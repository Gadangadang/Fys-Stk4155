from import_folders import *
import_all_folders()
from NeuralNetwork import NeuralNetwork
from NN_functions import *
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from plot_set import *
import numpy as np
import sys



if __name__ == "__main__":
    """Load breast cancer dataset"""

    np.random.seed(0)  # create same seed for random number every time
    cancer = load_breast_cancer()  # Download breast cancer dataset

    # Feature matrix of 569 rows (samples) and 30 columns (parameters)
    inputs = cancer.data
    # Label array of 569 rows (0 for benign and 1 for malignant)
    outputs = cancer.target

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
    cost_func = "cross_entropy"

    plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
    for i, activation in enumerate(activations):
        NN = NeuralNetwork(X_train,
                           y_train,
                           num_hidden_layers,
                           num_hidden_nodes,
                           batch_size,
                           0.01,
                           0.0,
                           gamma,
                           seed,
                           activation=activation,
                           cost="cross_entropy",
                           loss="accuracy",
                           callback=False,
                           last_activation="sigmoid")
        NN.train_network_stochastic(int(epochs))
        epoch = len(NN.score)
        epochs_lin = np.linspace(0, epoch, epoch)
        plt.plot(epochs_lin,
                 NN.score, label=labels[i], linestyle="-",
                 marker="None", markersize=3)
    plt.xlabel("epoch", fontsize=14)
    plt.ylim([0.92, 1])
    plt.ylabel("Accuracy", fontsize=14)
    plt.title("Activation function comparison for Breast cancer")
    plt.legend(fontsize=13)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    plt.show()
