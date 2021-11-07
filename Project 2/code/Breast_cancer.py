import numpy as np
import os
import sys
from NeuralNetwork import NeuralNetwork
from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from test_NN import sklearn_NN_WIP
from sklearn.preprocessing import StandardScaler
from SGD import SGD


def find_hyperparameters(X,
                         z,
                         epochs,
                         num_hidden_layers,
                         num_hidden_nodes,
                         batch_size,
                         etas,
                         lmbds,
                         activation,
                         cost_func,
                         name,
                         return_best = False
                         ):
    from sklearn.model_selection import train_test_split
    import seaborn as sns
    import matplotlib.ticker as tkr
    import matplotlib.pyplot as plt
    scaler = StandardScaler()

    sns.set()
    #Split and scale data
    X_train, X_test, Z_train, Z_test = train_test_split(X, z, test_size=0.2)
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    scaler.fit(X_test)
    X_test = scaler.transform(X_test)

    #Setup arrays for accuracy score
    test_accuracy = np.zeros((len(etas), len(lmbds)))

    for i, eta in enumerate(etas):
        for j, lmbd in enumerate(lmbds):
            NN = NeuralNetwork( X_train,
                                Z_train,
                                num_hidden_layers,
                                num_hidden_nodes,
                                batch_size,
                                eta,
                                lmbd,
                                activation = activation,
                                cost = cost_func)
            NN.train_network_stochastic(int(epochs))

            test_accuracy[i][j] = NN.accuracy_score(X_test, Z_test)

    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(test_accuracy, xticklabels = np.log10(lmbds), yticklabels = np.log10(etas), annot=True, ax=ax, cmap="viridis")
    ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda y, p: f'{y:.1f}'))
    ax.xaxis.set_major_formatter(tkr.FuncFormatter(lambda x, p: f'{x:.1f}'))
    ax.set_title("Test Accuracy")
    ax.set_ylabel("$log_{10}(\eta)$")
    ax.set_xlabel("$log_{10}(\lambda)$")
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    plt.subplots_adjust(hspace=0.3)
    #plt.savefig(f"../article/figures/hyper_param_{name}_{activation}.pdf",
    #            bbox_inches="tight")
    plt.show()
    if return_best:
        indx = np.where(test_accuracy == np.max(test_accuracy))
        return etas[int(indx[0][0])], lmbds[int(indx[1][0])], np.max(test_accuracy)




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
    eta = 1e-2
    lmbd = 0.5
    epochs = int(50)
    batch_size = int(100)

    etas = np.logspace(-4, 1, 5)
    lmbds = np.logspace(-4, 1, 5)

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
