import numpy as np
import os
import sys
from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from test_NN import sklearn_NN_WIP
from SGD import SGD
from find_hyperparameters import *


if __name__ == "__main__":
    from Functions import *
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

    num_hidden_layers = 2
    num_hidden_nodes = 30
    n_categories = 1
    epochs = int(100)
    batch_size = int(50)

    etas = np.logspace(-2, 0, 5)
    lmbds = np.logspace(-4, 0, 5)


    activation = "sigmoid"
    cost_func = "cross_entropy"
    name="breast_cancer"
    best_eta, best_lmbd, best_val = find_hyperparameters(X,
                                                         y,
                                                         epochs,
                                                         num_hidden_layers,
                                                         num_hidden_nodes,
                                                         batch_size,
                                                         etas,
                                                         lmbds,
                                                         activation,
                                                         cost_func,
                                                         name,
                                                         return_best = True)
    sklearn_pred, sklearn_accuracy = sklearn_NN_WIP(X,
                                                y,
                                                best_eta,
                                                best_lmbd,
                                                epochs,
                                                num_hidden_layers,
                                                num_hidden_nodes,
                                                n_categories,
                                                'logistic')
    print(f"{100*best_val:.0f}% NN accuracy. ")
    print(f"{100*sklearn_accuracy:.0f}% Sklearn accuracy")
