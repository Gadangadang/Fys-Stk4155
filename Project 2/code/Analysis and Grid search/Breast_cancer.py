import numpy as np
from import_folders import *
import_all_folders()
from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from test_NN import sklearn_NN_WIP
from SGD import SGD
from NN_functions import *
from NeuralNetwork import NeuralNetwork


if __name__ == "__main__":
    """Load breast cancer dataset"""

    np.random.seed(0)        #create same seed for random number every time
    cancer=load_breast_cancer()      #Download breast cancer dataset

    inputs=cancer.data                     #Feature matrix of 569 rows (samples) and 30 columns (parameters)
    outputs= cancer.target                #Label array of 569 rows (0 for benign and 1 for malignant)
    labels=cancer.feature_names[0:30]

    x = inputs
    y = outputs.reshape((len(outputs), 1))

    #Sampling only certain features.
    X = np.reshape(x[:,1],(len(x[:,1]),1))
    features = range(1, 30)
    for i in features:
        temp = np.reshape(x[:,i],(len(x[:,i]),1))
        X=np.hstack((X,temp))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_test = standard_scale(X_train, X_test)

    num_hidden_layers = 2
    num_hidden_nodes = 30
    batch_size = int(50)
    gamma = 0
    seed = 4155
    n_categories = 1
    epochs = int(200)
    batch_size = int(25)


    etas = np.logspace(-3, -1, 5)
    lmbds = np.logspace(-4, 0, 5)


    activation = "sigmoid"
    cost_func = "cross_entropy"
    test_scores = np.zeros((len(etas), len(lmbds)))

    NN = NeuralNetwork(X_train,
                       y_train,
                       num_hidden_layers,
                       num_hidden_nodes,
                       batch_size,
                       0.01,
                       0.0,
                       gamma,
                       seed,
                       activation = "relu",
                       cost = "cross_entropy",
                       loss = "accuracy",
                       callback = False,
                       last_activation = "sigmoid")
    NN.train_network_stochastic(int(100))
    NN.plot_score_history()
    
    for i, eta in enumerate(etas):
        for j, lmbd in enumerate(lmbds):
            print(f"\r(eta_val, lmbd_val) = ({i},{j})/({len(etas)-1},{len(lmbds)-1})", end="")
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
                               callback = True)
            NN.train_network_stochastic(int(epochs))
            test_scores[i][j] = NN.get_score(X_test, y_test)

    plot_heatmap(test_scores,[r"log($\eta$)",np.log10(etas)], [r"log($\lambda$)",np.log10(lmbds)], title = "Neural network test accuracy for Breast cancer data", name = None)
    indx = np.where(test_scores == np.max(test_scores))

    sklearn_pred, sklearn_accuracy = sklearn_NN_WIP(X,
                                                y,
                                                etas[indx[0][0]],
                                                lmbds[indx[1][0]],
                                                epochs,
                                                num_hidden_layers,
                                                num_hidden_nodes,
                                                n_categories,
                                                'logistic')

    print(f"{100*test_scores[indx[0][0]][indx[1][0]]:.0f}% NN accuracy. ")
    print(f"{100*sklearn_accuracy:.0f}% Sklearn accuracy")
