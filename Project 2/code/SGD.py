import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from tqdm import trange
import matplotlib.pyplot as plt


def SGD(X, y, eta_val=0.1):
    """
    Stochastic Gradient Descent
    with mini batches
    """

    dim_check = X.shape[0] == y.shape[0]
    assert dim_check, "Dimensions of X and y does not match"

    N = X.shape[0]                      # Number of data points
    m = 10                              # Size of each mini-batch
    num_epochs = int(1e4)                 # Number of epochs
    theta = np.ones((X.shape[1], 1))    # Initialize theta with ones
    def eta(epoch): return eta_val            # Learning rate (constant here)
    np.random.seed(4155)

    for epoch in range(1, num_epochs + 1):
        batch = np.random.choice(N, m, replace=False)  # Mini batch
        g = 2.0 / len(y) * X.T @ ((X @ theta) - y)  # Compute gradient
        #print(g, theta)
        theta -= eta(epoch) * g  # Update theta

    return theta.ravel()


def compare_SGD_OLS(X, z, eta):
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

    # Scale bu subtracting mean
    mean_scale(X_train, X_test, z_train, z_test)

    # OLS regression
    beta_OLS = OLS_regression(X_train, z_train)
    ztilde = (X_train @ beta_OLS).ravel()
    zpredict = (X_test @ beta_OLS).ravel()

    # MSE for OLS
    MSE_train = MSE(z_train, ztilde)
    MSE_test = MSE(z_test, zpredict)
    print("--- OLS ---")
    print("train err {:.7f} test err {:.7f}".format(MSE_train, MSE_test))

    OLS_MSE_train = np.ones(len(eta)) * MSE_train
    OLS_MSE_test = np.ones(len(eta)) * MSE_test

    # MSE for SGD
    SGD_MSE_train = np.zeros(len(eta))
    SGD_MSE_test = np.zeros(len(eta))

    print("--- SGD ---")
    for index, eta_val in enumerate(eta):
        # Find theta
        theta_train = SGD(X_train, z_train, eta_val=eta_val)
        theta_test = SGD(X_test, z_test, eta_val=eta_val)

        # Prediction
        ztilde_theta = (X_train @ theta_train).ravel()
        zpredict_theta = (X_test @ theta_test).ravel()

        # Error
        train_err = MSE(ztilde_theta, z_train)
        test_err = MSE(zpredict_theta, z_test)

        print("train err {:.7f} test err {:.7f}".format(train_err, test_err))

        SGD_MSE_train[index] = train_err
        SGD_MSE_test[index] = test_err

    plotting(eta, OLS_MSE_train, SGD_MSE_train)
    plotting(eta, OLS_MSE_test, SGD_MSE_test)


def plotting(eta, OLS_MSE, SGD_MSE):
    plt.plot(eta, OLS_MSE, label="OLS")
    plt.plot(eta, SGD_MSE, label="SGD")
    plt.xlabel(r"$\eta - Learning rate$")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Get modules from project 1
    path = os.getcwd()  # Current working directory
    path += '/../../Project 1/code'
    sys.path.append(path)
    from Functions import *

    #--- Create data from Franke Function ---#
    N = 10               # Number of points in each dimension
    z_noise = 0.2       # Added noise to the z-value
    n = 2               # Highest order of polynomial for X
    lamda = 0
    x, y, z = generate_data(N, z_noise)
    X = create_X(x, y, n)
    # X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
    # mean_scale(X_train, X_test, z_train, z_test)
    eta = np.linspace(0.001, 0.5, 15)

    #--- Regression ---#
    theta = SGD(X, z)  # Stochastic Gradient Descent
    theta_OLS = OLS_regression(X, z)  # OLS regression

    print("SGD: ", theta)
    print("OLS:", theta_OLS)

    #--- Comparison SGD OLS ---#
    compare_SGD_OLS(X, z, eta)
