import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from Functions import *

def Error_Complexity(N, z_noise, n, plot = True, Return = False):
    error_test, error_train = np.zeros(n+1), np.zeros(n+1)
    x, y, z = generate_data(N, z_noise)
    z = standard_scale(z)
    for i in range(0,n):
        X = create_X(x, y, i)
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        beta_OLS = OLS_regression(X_train, z_train)

        ztilde = X_train @ beta_OLS
        zpredict = X_test @ beta_OLS

        error_test[i] = (MSE(z_test,zpredict))
        error_train[i] = (MSE(z_train,ztilde))

    if Return:
        return error_test, error_train
    if plot:
        import matplotlib.pyplot as plt
        n_arr = np.linspace(0,n,n+1)
        plt.plot(n_arr, error_test, label = "Test")
        plt.plot(n_arr, error_train, label = "Train")
        plt.legend()
        plt.show()
    return


if __name__ == "__main__":
    N = 400             # Number of points in each dimension
    z_noise = 1      # Added noise to the z-value
    n = 30              # Highest order of polynomial for X

    Error_Complexity(N, z_noise,n)
