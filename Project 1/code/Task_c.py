import numpy as np
import matplotlib.pyplot as plt
from random import random, seed
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from Functions import *
from Task_b2 import bootstrap
from matplotlib.ticker import MaxNLocator


def cross_validation(X, z, k_fold_number, method, lamda=0, include_train=False):
    kfold = KFold(n_splits=k_fold_number)
    j = 0
    z_pred_arr = np.zeros((int(np.shape(X)[0] / k_fold_number), k_fold_number))

    MSE_arr = np.zeros(k_fold_number)
    if include_train:
        MSE_arr_tilde = np.zeros(k_fold_number)

    for train_indx, test_indx in kfold.split(X):
        X_train = X[train_indx]
        z_train = z[train_indx]

        X_test = X[test_indx]
        z_test = z[test_indx]

        X_train, X_test = scale_design_matrix(X_train, X_test)
        if method == "OLS":
            beta = OLS_regression(X_train, z_train)
            if include_train:
                z_tilde = (X_train @ beta).ravel()

        elif method == "Ridge":
            beta = RIDGE_regression(X_train, z_train, lamda)
            if include_train:
                z_tilde = (X_train @ beta).ravel()

        z_pred = (X_test @ beta).ravel()
        z_pred_arr[:, j] = z_pred

        MSE_arr[j] = MSE(z_test.ravel(), z_pred)
        if include_train:
            MSE_arr_tilde[j] = MSE(z_train.ravel(), z_tilde)

        j += 1
    if include_train:
        return np.mean(MSE_arr), np.mean(MSE_arr_tilde)
    return np.mean(MSE_arr)


def compaire_CV_B(N, z_noise, n, B, k_fold_number, method):
    x, y, z = generate_data(N, z_noise, seed=2018)

    error_CV = np.zeros(n + 1)
    error_B = np.zeros(n + 1)
    error_sklearn = np.zeros(n + 1)
    ols = LinearRegression(fit_intercept=False)
    for i in range(0, n + 1):  # For increasing complexity
        X = create_X(x, y, i)

        X_train, X_test, z_train, z_test = train_test_split(
            X, z, test_size=0.2)

        X_train, X_test = scale_design_matrix(X_train, X_test)

        z_pred_B = bootstrap(X_train, X_test, z_train, z_test, B, method)

        error_CV[i] = cross_validation(X, z, k_fold_number, method)
        error_B[i] = np.mean(
            np.mean((z_test - z_pred_B)**2, axis=1, keepdims=True))
        error_sklearn[i] = np.mean(-cross_val_score(ols, X, z,
                                                    scoring='neg_mean_squared_error', cv=k_fold_number))

    n_arr = np.linspace(0, n, n + 1)

    plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
    plt.plot(n_arr, error_CV, label="Cross validation ")
    plt.plot(n_arr, error_B, label="Bootstrap")
    plt.plot(n_arr, error_sklearn, "--", label="sklearn")

    ax = plt.gca()
    # Force integer ticks on x-axis
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel(r"$n [complexity]$", fontsize=14)
    plt.ylabel(r"MSE", fontsize=14)
    plt.title(r"Bootstrap iterations: {}, K-folds: {}".format(B,
                                                              k_fold_number), fontsize=16)
    plt.legend(fontsize=13)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    plt.show()


if __name__ == "__main__":
    N = 30
    z_noise = 0.1
    n = 10
    B = 100
    k_fold_number = 10
    method = "OLS"
    compaire_CV_B(N, z_noise, n, B, k_fold_number, method)
