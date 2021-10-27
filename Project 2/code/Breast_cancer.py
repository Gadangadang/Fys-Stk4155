import numpy as np
import os
import sys
from NeuralNetwork import NeuralNetwork
from sklearn.datasets import load_breast_cancer


if __name__ == "__main__":
    # Get modules from project 1
    path = os.getcwd()  # Current working directory
    path += '/../../Project 1/code'
    sys.path.append(path)
    from Functions import *

    """Load breast cancer dataset"""

    np.random.seed(0)        #create same seed for random number every time

    cancer=load_breast_cancer()      #Download breast cancer dataset

    inputs=cancer.data                     #Feature matrix of 569 rows (samples) and 30 columns (parameters)
    outputs= cancer.target                #Label array of 569 rows (0 for benign and 1 for malignant)
    labels=cancer.feature_names[0:30]
    x = inputs
    y = outputs.reshape((len(outputs), 1))
    temp1=np.reshape(x[:,1],(len(x[:,1]),1))
    temp2=np.reshape(x[:,2],(len(x[:,2]),1))
    X=np.hstack((temp1,temp2))
    temp=np.reshape(x[:,5],(len(x[:,5]),1))
    X=np.hstack((X,temp))
    temp=np.reshape(x[:,8],(len(x[:,8]),1))
    X=np.hstack((X,temp))
    num_hidden_layers = 2
    num_hidden_nodes = 10
    n_categories = 1
    eta = 1e-2
    lmbd = 0.01
    epochs = int(1e4)
    batch_size = int(len(X)/4)
    NN = NeuralNetwork( X,
                        y,
                        num_hidden_layers,
                        num_hidden_nodes,
                        batch_size,
                        eta,
                        lmbd,
                        4155,
                        "sigmoid",
                        "binary_difference")
    NN.run_network_stochastic(int(epochs))
    print(NN.layers_a[-1])

    accuracy = NN.accuracy_score()
    print(f"{int(accuracy*len(NN.t))} / {len(NN.t)} ({100*accuracy:.0f}%) accurate predictions. ")
