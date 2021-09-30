import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from Functions import *
from matplotlib.ticker import MaxNLocator
from sklearn.utils import resample
from plot_set import *  # Specifies plotting settings


def bias_variance_tradeoff_lamda(lamda_values, N, z_noise, n, B, method, plot=True):
    """Calculates the bias variance tradeoff with a given
       linear regression model. Then generates a plot showing
       the tradeoff.

    Args:
        lamda_values (Array): List of floats with adjustment parameter lamda. Defaults to [0].
        N              (Int): Dimension on the datasets
        z_noise      (Float): Noise scalar, scales the normally distributed noise
        n              (Int): Highest order complexity
        B              (Int): Amount of bootstrap iterations
        method      (String): Choice of linear regression model
        plot(bool, optional): Choice to produce plot or not. Defaults to True.


    """
    x, y, z = generate_data(N, z_noise, seed=4155)
    bias = np.zeros(n + 1)
    variance = np.zeros(n + 1)
    error = np.zeros(n + 1)

    for lamda in lamda_values:
        for i in range(0, n + 1):  # For increasing complexity

            X = create_X(x, y, i)

            X_train, X_test, z_train, z_test = train_test_split(
                X, z, test_size=0.2)
            mean_scale(X_train, X_test, z_train, z_test)

            z_pred, z_tilde = bootstrap(
                X_train, X_test, z_train, z_test, B, method, lamda, include_train=True)
            # axis = 1 => columns
            bias[i] = np.mean(
                (z_test - np.mean(z_pred, axis=1, keepdims=True))**2)
            variance[i] = np.mean(np.var(z_pred, axis=1))
            error[i] = np.mean(
                np.mean((z_test - z_pred)**2, axis=1, keepdims=True))

        n_arr = np.linspace(0, n, n + 1)
        if plot:
            plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
            plt.plot(n_arr, bias, label="Bias")
            plt.plot(n_arr, variance, label="Variance")
            plt.plot(n_arr, error, "--", label="Error")
            plt.title(r"${}: \lambda$ = {:.3f}".format(method, lamda))
            ax = plt.gca()
            # Force integer ticks on x-axis
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.xlabel(r"$n$", fontsize=14)
            # plt.ylabel(r"MSE", fontsize=14)
            plt.legend(fontsize=13)
            plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
            # plt.savefig(f"../article/figures/bias_variance_tradeoff.pdf", bbox_inches="tight")
            plt.show()


def Error_Complexity_boot(lamda, N, z_noise, n, seed=4155):
    """Computes the error for bootstrap

    Args:
        lamda (Float): Lamda value in logspace
        N (Int): Dimension of datasets
        z_noise (Float): Noise scalar, used to scale the normally distributed noise
        n (Int): Highest order complexity for design matrix
        seed (int, optional): Random seed. Defaults to 4155.

    Returns:
        Array: Error for train
        Array: Error for test
    """
    error_test, error_train = np.zeros(n + 1), np.zeros(n + 1)
    x, y, z = generate_data(N, z_noise, seed)

    for i in range(0, n + 1):
        X = create_X(x, y, i)
        X_train, X_test, z_train, z_test = train_test_split(
            X, z, test_size=0.2)

        mean_scale(X_train, X_test, z_train, z_test)

        z_pred, z_tilde = bootstrap(
            X_train, X_test, z_train, z_test, B, "Ridge", lamda, include_train=True)
        error_test[i] = np.mean(
            np.mean((z_test - z_pred)**2, axis=1, keepdims=True))
        error_train[i] = np.mean(
            np.mean((z_train - z_tilde)**2, axis=1, keepdims=True))

    return error_test, error_train


def Error_Complexity_CV(lamda, N, z_noise, n, k_fold_number, seed=4155):
    """Computes the error for using cross validation to find the optimal model

    Args:
        lamda         (Float): Lamda value in logscale
        N               (Int): Dimension of data sets
        z_noise       (Float): Noise scalar, used to scale the normally distributed noise
        n               (Int): Highest order complexity of design matrix
        k_fold_number   (Int): Number of k fold for cross validation
        seed  (int, optional): Random seed. Defaults to 4155.

    Returns:
        Array: Error for the test case
        Array: Error for the train case
    """
    error_test, error_train = np.zeros(n + 1), np.zeros(n + 1)
    x, y, z = generate_data(N, z_noise, seed)


    for i in range(0, n + 1):
        X = create_X(x, y, i)

        error_test[i], error_train[i] = cross_validation(
            X, z, k_fold_number, "Ridge", lamda, include_train=True)

    return error_test, error_train


def MSE_Ridge_bootstrap(N, z_noise, n, lamda_values):
    """Calculates the mean squared error for ridge regression with bootstrap method

    Args:
        N              (Int): Dimension of data sets
        z_noise      (Float): Noise scalar, used to scale the normally distributed noise
        n              (Int): Highest order complexity of design matrix
        lamda_values (Array): Array with lamda values, in logspace
    """
    plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')

    for index, lamda in enumerate(lamda_values):

        plt.subplot(2, 2, index + 1)
        error_test, error_train = Error_Complexity_boot(
            lamda, N, z_noise, n,  seed=4155)
        n_arr = np.linspace(0, n, n + 1)

        #---Plotting---#

        plt.title(r"Bootstrap MSE with $\lambda$ = {:.3f} ".format(lamda))
        plt.plot(n_arr, error_test, label="Test")
        plt.plot(n_arr, error_train, label="Train")
        plt.legend(fontsize=13)

    ax = plt.gca()
    # Force integer ticks on x-axis
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel(r"$n$", fontsize=14)
    plt.ylabel(r"MSE", fontsize=14)

    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    # plt.savefig(f"../article/figures/Complexity_MSE{numRuns}.pdf", bbox_inches="tight")
    plt.show()


def MSE_Ridge_CV(N, z_noise, n, lamda_values, k_fold_number):
    """[summary]

    Args:
        N              (Int): Dimension on the data sets
        z_noise      (Float): Noise scalar, used to scale the normally distributed noise
        n              (Int): Highest order complexity
        lamda_values (Array): Array of lamda values in logspace
        k_fold_number  (Int): Number of k fold for cross validation
    """
    plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')

    for index, lamda in enumerate(lamda_values):

        plt.subplot(2, 2, index + 1)
        error_test, error_train = Error_Complexity_CV(
            lamda, N, z_noise, n, k_fold_number, seed=4155)
        n_arr = np.linspace(0, n, n + 1)

        print("Minimum value for train is {:.3f} with lambda = {:.3f}".format(
            np.min(error_test), lamda))
        #---Plotting---#

        plt.title(r"CV MSE with $\lambda$ = {:.3f} ".format(lamda))
        plt.plot(n_arr, error_test, label="Test")
        plt.plot(n_arr, error_train, label="Train")
        plt.legend(fontsize=13)

    ax = plt.gca()
    # Force integer ticks on x-axis
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel(r"$n$", fontsize=14)
    plt.ylabel(r"MSE", fontsize=14)

    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    # plt.savefig(f"../article/figures/Complexity_MSE{numRuns}.pdf", bbox_inches="tight")
    plt.show()


def lamdaDependency(N, z_noise, n, lamda):
    """Computes the dependency of lamda for the beta values

    Args:
        N         (Int): Dimension for the datasets
        z_noise (Float): Noise scalar, used to scale the normally distributed noise
        n         (Int): Highest order complexity for design matrix
        lamda   (Array): Array of lamda values in logscale
    """
    x, y, z = generate_data(N, z_noise, seed=2018)
    X = create_X(x, y, n)
    X_train, X_test, z_train, z_test = train_test_split(
        X, z, test_size=0.2)
    mean_scale(X_train, X_test, z_train, z_test)
    length = len(RIDGE_regression(X_train, z_train, 0.1))
    beta_R = np.zeros((int(length), len(lamda)))

    i = 0
    for lmb in lamda:
        beta_R[:, i] = RIDGE_regression(X_train, z_train, lmb).ravel()
        i += 1
    for j in range(len(lamda)):
        plt.plot((lamda), beta_R[j, :])
    plt.ylabel(r"$\beta$", size=16)
    plt.xlabel(r"$\lambda$", size=16)
    plt.xscale('log')
    plt.title(
        r"$\beta_i (\lambda)$ - [degree = {} and N = {}] ".format(n, N), size=16)
    plt.show()


if __name__ == "__main__":
    #--- settings ---#
    N = 30           # Number of points in each dimension
    z_noise = 0.2     # Added noise to the z-value
    n = 14                 # Highest order of polynomial for X
    B = 100           # Number of iterations in boostrap
    k_fold_number = 10
    # Bootstrap analysis with Ridge
    lamda_values = np.logspace(-3, 2, 4)
    method = "Ridge"
    #MSE_Ridge_bootstrap(N, z_noise, n, lamda_values)

    # Cross-validation with Ridge
    MSE_Ridge_CV(N, z_noise, n, lamda_values, k_fold_number)
    # Bias-variance tradeoff with Ridge

    #bias_variance_tradeoff_lamda(lamda_values, N, z_noise, n, B, method, plot = True)

    #lamdaDependency(22, 0.2, 15, np.logspace(-5, 0, 20))
