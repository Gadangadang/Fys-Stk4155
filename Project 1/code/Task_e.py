import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from Functions import *
from matplotlib.ticker import MaxNLocator
from sklearn.utils import resample
from plot_set import *  # Specifies plotting settings
from Task_b2 import bias_variance_tradeoff
from Task_c import compaire_CV_B
from Task_d import MSE_bootstrap


def complexity_CV(data, n, k_fold_number, method, lamda=0., plot=True, seed=4155):
    """Compares train and test MSE using CV for Lasso method

    Args:
        data            (Array): 1D array with x,y,z data
        n                 (Int): Number for highest order complexity
        k_fold_number     (Int): Number of folds for CV
        method         (String): Choice of method
        lamda (Float, optional): Float value, tuning parameter, could also be array. Defaults to 0.
        plot   (bool, optional): Choice if plot or not. Defaults to True.
        seed    (int, optional): Choice of seed. Defaults to 4155.
    """
    k = 0
    x, y, z = data
    for lmb in lamda:
        MSE_test, MSE_train = np.zeros(n + 1), np.zeros(n + 1)
        for i in range(0, n + 1):
            X = create_X(x, y, i)
            MSE_test[i], MSE_train[i] = cross_validation(
                X, z, k_fold_number, method, lmb, include_train=True)

        if plot:
            #---Plotting---#
            n_arr = np.linspace(0, n, n + 1)

            plt.figure(num=0, figsize=(8, 6),
                       facecolor='w', edgecolor='k')
            plt.subplot(2, 2, k + 1)
            if method == "OLS":
                plt.title(f"{method}: N = {N}", size=14)
            else:
                plt.title(
                    f"{method} $\lambda = ${lmb:.4f} : N = {N}", size=14)
            plt.plot(n_arr[1:], MSE_train[1:], label="Train")
            plt.plot(n_arr[1:], MSE_test[1:], label="Test")
            ax = plt.gca()
            # Force integer ticks on x-axis
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.xlabel(r"$n$", fontsize=14)
            plt.ylabel(r"MSE", fontsize=14)

            # plt.ylabel(r"MSE", fontsize=14)
            plt.tight_layout()
            if k == 0:
                plt.legend(fontsize=13)

        k += 1
    plt.show()


def complexity_boot(data, n, method, B, lamda=0, plot=True, seed=4155):
    """Compares train and test MSE using CV for Lasso method

    Args:
        data            (Array): 1D array with x,y,z data
        n                 (Int): Number for highest order complexity
        method         (String): Choice of method
        B                 (Int): Number of bootstrap iterations
        lamda (Float, optional): Float value, tuning parameter, could also be array. Defaults to 0.
        plot   (bool, optional): Choice if plot or not. Defaults to True.
        seed    (int, optional): Choice of seed. Defaults to 4155.
    """
    k = 0
    x, y, z = data
    for lmb in lamda:
        MSE_test, MSE_train = np.zeros(n + 1), np.zeros(n + 1)
        for i in range(0, n + 1):
            X = create_X(x, y, i)
            X_train, X_test, z_train, z_test = train_test_split(
                X, z, test_size=0.2)

            mean_scale(X_train, X_test, z_train, z_test)
            z_pred, z_tilde = bootstrap(
                X_train, X_test, z_train, z_test, B, method, lamda=lmb, include_train=True)

            MSE_test[i] = np.mean(
                np.mean((z_test - z_pred)**2, axis=1, keepdims=True))
            MSE_train[i] = np.mean(
                np.mean((z_train - z_tilde)**2, axis=1, keepdims=True))
        if plot:
            #---Plotting---#
            n_arr = np.linspace(0, n, n + 1)

            plt.figure(num=0, figsize=(8, 6),
                       facecolor='w', edgecolor='k')
            plt.subplot(2, 2, k + 1)
            if method == "OLS":
                plt.title(f"{method}: N = {N}", size=14)
            else:
                plt.title(
                    f"{method} $\lambda = ${lmb:.4f} : N = {N}", size=14)
            plt.plot(n_arr[1:], MSE_train[1:], label="Train")
            plt.plot(n_arr[1:], MSE_test[1:], label="Test")
            ax = plt.gca()
            # Force integer ticks on x-axis
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.xlabel(r"$n$", fontsize=14)
            plt.ylabel(r"MSE", fontsize=14)

            # plt.ylabel(r"MSE", fontsize=14)
            plt.tight_layout()
            if k == 0:
                plt.legend(fontsize=13)

        k += 1
    plt.show()


if __name__ == "__main__":
    #--- settings ---#
    N = 15          # Number of points in each dimension
    z_noise = 0.2     # Added noise to the z-value
    n = 24                # Highest order of polynomial for X
    B = "N"            # Number of training points
    method = "Lasso"
    lamda = np.logspace(-3, -1, 4)
    k_fold_number = 5
    B = 100

    #bias_variance_tradeoff(N, z_noise, n, "N", method, lamda, plot=True)

    #compaire_CV_B(N, z_noise, n, N * N, k_fold_number, method, lamda=1e-3)
    #complexity_CV(generate_data(N, z_noise, seed=4155), n, k_fold_number, method, lamda, plot=True, seed=4155)
    N = 30
    z_noise = 0.2
    n = 14
    B = 100

    method = "Lasso"

    lmb = 1e-4
    compaire_CV_B(generate_data(N, z_noise, seed=4155), n, B, k_fold_number, method, lamda = lmb)

    lmb = 1e-3
    compaire_CV_B(generate_data(N, z_noise, seed=4155), n, B, k_fold_number, method, lamda = lmb)

    lmb = 1e-2
    compaire_CV_B(generate_data(N, z_noise, seed=4155), n, B, k_fold_number, method, lamda = lmb)

    lmb = 1e2
    compaire_CV_B(generate_data(N, z_noise, seed=4155), n, B, k_fold_number, method, lamda = lmb)
