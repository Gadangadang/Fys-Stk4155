import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from Functions import *
from matplotlib.ticker import MaxNLocator
from sklearn.utils import resample
from plot_set import * # Specifies plotting settings
from sklearn import linear_model


def bootstrap(X_train, X_test, z_train, z_test, B, method, lamda = 0, include_train = False):
    """
    info
    """

    z_pred = np.zeros((len(z_test), B))
    if include_train:
        z_tilde = np.zeros((len(z_train), B))
    if method == "OLS":
        for i in range(B):
            X_res, z_res = resample(X_train, z_train)
            beta = OLS_regression(X_res, z_res)
            z_pred[:,i] = (X_test @ beta).ravel()
            if include_train:
                z_tilde[:,i] = (X_train @ beta).ravel()
    elif method == "Ridge":
        for i in range(B):
            X_res, z_res = resample(X_train, z_train)
            beta = RIDGE_regression(X_res, z_res, lamda)
            z_pred[:,i] = (X_test @ beta).ravel()
            if include_train:
                z_tilde[:,i] = (X_train @ beta).ravel()
    elif method == "Lasso":
        for i in range(B):
            X_res, z_res = resample(X_train, z_train)
            RegLasso = linear_model.Lasso(lmb)
            RegLasso.fit(X_res,z_res)
            z_pred[:,i] = RegLasso.predict(X_test)
            if include_train:
                z_tilde[:,i] = RegLasso.predict(X_train)
    if include_train:
        return z_pred, z_tilde
    else:
        return z_pred





def bias_variance_tradeoff(N, z_noise, n, B, plot = True):
    """
    write info
    """
    x, y, z = generate_data(N, z_noise, seed=2018)
    bias = np.zeros(n+1)
    variance = np.zeros(n+1)
    error = np.zeros(n+1)

    for i in range(0,n+1): #For increasing complexity

        X = create_X(x, y, i)

        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

        X_train, X_test = scale_design_matrix(X_train, X_test)

        z_pred = bootstrap(X_train, X_test, z_train, z_test, B, "OLS")
        bias[i] = np.mean((z_test - np.mean(z_pred, axis = 1, keepdims = True))**2) # axis = 1 => columns
        variance[i] = np.mean(np.var(z_pred, axis = 1))
        error[i] = np.mean(np.mean( (z_test-z_pred)**2, axis = 1, keepdims = True  ))

    n_arr = np.linspace(0,n,n+1)
    if plot:
        plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
        plt.plot(n_arr, bias, label = "Bias")
        plt.plot(n_arr, variance, label = "Variance")
        plt.plot(n_arr, error, "--", label = "Error")

        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # Force integer ticks on x-axis
        plt.xlabel(r"$n$", fontsize=14)
        # plt.ylabel(r"MSE", fontsize=14)
        plt.legend(fontsize = 13)
        plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
        # plt.savefig(f"../article/figures/bias_variance_tradeoff.pdf", bbox_inches="tight")
        plt.show()







if __name__ == "__main__":
    #--- settings ---#
    N = 100            # Number of points in each dimension
    z_noise = 0.1     # Added noise to the z-value
    n = 14                 # Highest order of polynomial for X
    B = 100

    bias_variance_tradeoff(N, z_noise, n, B, plot = True)
