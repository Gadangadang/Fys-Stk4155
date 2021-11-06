import numpy as np
import os
import sys
from NeuralNetwork import NeuralNetwork
from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from test_NN import sklearn_NN
from sklearn.preprocessing import StandardScaler


if __name__ == "__main__":
    # Get modules from project 1
    path = os.getcwd()  # Current working directory
    path += '/../../Project 1/code'
    sys.path.append(path)
    from Functions import *
    scaler = StandardScaler()
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


    features = range(1, 30)#[2,5,8]
    for i in features:
        temp = np.reshape(x[:,i],(len(x[:,i]),1))
        X=np.hstack((X,temp))

    #Scaling of data


    X_train, X_test, Z_train, Z_test = train_test_split(X, y, test_size=0.2)
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)

    scaler.fit(X_test)
    X_test = scaler.transform(X_test)

    num_hidden_layers = 2
    num_hidden_nodes = 150
    n_categories = 1
    eta = 1e-2
    lmbd = 0.5
    epochs = int(6e3)
    batch_size = int(100)
    #X_train, X_test, Z_train, Z_test = train_test_split(X, z, test_size=0.2)
    sklearn_pred, sklearn_accuracy = sklearn_NN(X_train, Z_train.ravel(), eta, lmbd, epochs, num_hidden_layers, num_hidden_nodes, n_categories, 'logistic')
    NN = NeuralNetwork( X_train,
                        Z_train,
                        num_hidden_layers,
                        num_hidden_nodes,
                        batch_size,
                        eta,
                        lmbd,
                        4155,
                        "sigmoid",
                        "cross_entropy")
    NN.run_network_stochastic(int(epochs))
    #print(np.asarray(NN.layers_a[-1]).ravel())

    accuracy = NN.accuracy_score(X_test,Z_test)
    print(f"{100*accuracy:.0f}% NN accuracy -- {int(accuracy*len(NN.t))} / {len(NN.t)} accurate predictions. ")
    #print(f"{100*sklearn_accuracy:.0f}% Sklearn accuracy")
