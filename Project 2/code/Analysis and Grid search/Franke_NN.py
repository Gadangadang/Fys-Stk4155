import os
import sys
from sklearn.model_selection import train_test_split
from import_folders import *
import_all_folders()
sys.path.insert(1,"../../../Project 1/code/")
from Functions import *
from NeuralNetwork import *
# The above imports numpy as np so we have to redefine:
import autograd.numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
import seaborn as sns
import matplotlib.ticker as tkr
import matplotlib.pyplot as plt
sns.set()


if __name__ == "__main__":




    #--- Create data from Franke Function ---#
    N = 10               # Number of points in each dimension
    z_noise = 0.2       # Added noise to the z-value
    n = 8               # Highest order of polynomial for X
    epochs = 1000
    iterations = 1
    batch_size = int(N * N * 0.8)

    x, y, z = generate_data(N, z_noise)
    X = create_X(x, y, n)

    #pseudo code
    x = np.randnum(0,1, num_points)
    y = np.randnum(0,1, num_points)
    z = Franke(x,y + noise)
    X = [x, y]


    X_train, X_test, Z_train, Z_test = train_test_split(X, z, test_size=0.2)

    beta = OLS_regression(X_train, Z_train)
    z_ols = X_test @ beta


    etas = np.logspace(-4, 1, 10)
    lmbds = np.logspace(-4, 1, 10)




    MSE_accuracy = np.zeros((len(etas), len(lmbds)))
    R2_accuracy = np.zeros((len(etas), len(lmbds)))
    for i, eta in enumerate(etas):
        for j, lmbd in enumerate(lmbds):
            MM = NeuralNetwork(X_train,
                               Z_train,
                               num_hidden_layers=5,
                               num_hidden_nodes=10,
                               batch_size=batch_size,
                               eta=eta,
                               lmbd=lmbd,
                               seed=4155,
                               activation="sigmoid",
                               cost="MSE")

            MM.train_network_stochastic(epochs)
            prediction = MM.predict(X_test)
            MSE_accuracy[i, j] = MSE(Z_test, prediction)
            R2_accuracy[i, j] = R2(Z_test, prediction)

    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(MSE_accuracy, xticklabels = np.log10(lmbds), yticklabels = np.log10(etas), annot=True, ax=ax, cmap="viridis")
    ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda y, p: f'{y:.1f}'))
    ax.xaxis.set_major_formatter(tkr.FuncFormatter(lambda x, p: f'{x:.1f}'))
    ax.set_title(f"Test Accuracy with sigmoid")
    ax.set_ylabel("$log_{10}(\eta)$")
    ax.set_xlabel("$log_{10}(\lambda)$")
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    plt.subplots_adjust(hspace=0.3)
    plt.savefig(f"../article/figures/hyper_param_MSE_sigmoid.pdf",
                bbox_inches="tight")
    plt.show()

    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(R2_accuracy, xticklabels = np.log10(lmbds), yticklabels = np.log10(etas), annot=True, ax=ax, cmap="viridis")
    ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda y, p: f'{y:.1f}'))
    ax.xaxis.set_major_formatter(tkr.FuncFormatter(lambda x, p: f'{x:.1f}'))
    ax.set_title(f"Test Accuracy with sigmoid")
    ax.set_ylabel("$log_{10}(\eta)$")
    ax.set_xlabel("$log_{10}(\lambda)$")
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    plt.subplots_adjust(hspace=0.3)
    plt.savefig(f"../article/figures/hyper_param_R2_sigmoid.pdf",
                bbox_inches="tight")
    plt.show()

    print(f"Neural Network SGD sigmoid", MSE(Z_test, MM.predict(X_test)))

    print("            OLS            ", MSE(Z_test, z_ols))
    """
    MM = NeuralNetwork(X_train,
                       Z_train,
                       num_hidden_layers=5,
                       num_hidden_nodes=10,
                       batch_size=batch_size,
                       eta=,
                       lmbd=lmbd,
                       seed=4155,
                       activation=relu,
                       cost="MSE")

    MM.train_network_stochastic(epochs)
    prediction = MM.predict(X_test)
    """
