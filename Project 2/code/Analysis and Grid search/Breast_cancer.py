from import_folders import *
import_all_folders()
from NeuralNetwork import NeuralNetwork
from NN_functions import *
from SGD import SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
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

    num_hidden_layers = 3
    num_hidden_nodes = 40
    batch_size = int(50)
    gamma = 0
    seed = 4155
    n_categories = 1

    epochs = int(50)  # int(200)
    batch_size = int(25)
    etas = np.logspace(-3, -1, 5)
    lmbds = np.logspace(-4, 0, 5)

    activation = "sigmoid"
    cost_func = "cross_entropy"
    test_scores = np.zeros((len(etas), len(lmbds)))

    for i, eta in enumerate(etas):
        for j, lmbd in enumerate(lmbds):
            print(
                f"\r(eta_val, lmbd_val) = ({i},{j})/({len(etas)-1},{len(lmbds)-1})", end="")
            NN = NeuralNetwork(X_train,
                               y_train,
                               num_hidden_layers,
                               num_hidden_nodes,
                               batch_size,
                               eta,
                               lmbd,
                               gamma,
                               seed,
                               activation,
                               cost="cross_entropy",
                               loss = "accuracy",
                               callback = False)

            NN.train_network_stochastic(int(epochs))
            NN.train_network_stochastic(int(epochs))
            # NN.set_eta_decay(k=0.1, dropp_time = 10) Remove hastag for eta decay.
            test_scores[i][j] = NN.get_score(X_test, y_test)

    plot_heatmap(test_scores, [r"log($\lambda$)", np.log10(lmbds)], [r"log($\eta$)", np.log10(
        etas)], title="Neural network test accuracy for Breast cancer data", name=None)
    indx = np.where(test_scores == np.max(test_scores))

    sklearn_pred, sklearn_accuracy = sklearn_NN(X,
                                                    y,
                                                    etas[indx[0][0]],
                                                    lmbds[indx[1][0]],
                                                    epochs,
                                                    num_hidden_layers,
                                                    num_hidden_nodes,
                                                    n_categories,
                                                    'logistic')

    print(f"{100*test_scores[indx[0][0]][indx[1][0]]:.2f}% NN accuracy. ")
    print(f"{100*sklearn_accuracy:.2f}% Sklearn accuracy")
