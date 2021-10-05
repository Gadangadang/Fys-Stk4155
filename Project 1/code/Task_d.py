import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from Functions import *
from matplotlib.ticker import MaxNLocator
from sklearn.utils import resample
from plot_set import *  # Specifies plotting settings
from Task_c import compaire_CV_B
from Task_b2 import bias_variance_tradeoff


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
    plt.ylabel(r"$\beta_i$", size=16)
    plt.xlabel(r"$\lambda$", size=16)
    plt.xscale('log')
    plt.title(
        r"$\beta _i (\lambda)$ - [degree = {} and N = {}] ".format(n, N), size=16)
    plt.show()


if __name__ == "__main__":
    #--- settings ---#
    N = 15           # Number of points in each dimension
    z_noise = 0.2     # Added noise to the z-value
    n = 20                 # Highest order of polynomial for X
    B = "N"           # Number of iterations in boostrap
    k_fold_number = 10
    # Bootstrap analysis with Ridge
    lamda_values = np.logspace(-5, -2, 4)
    method = "Ridge"
    # Bias-variance tradeoff with Ridge
    bias_variance_tradeoff(N, z_noise, n, B, method, lamda_values, plot=True)

    lamdaDependency(22, 0.2, 15, np.logspace(-5, 0, 20))
    N = 30
    z_noise = 0.2
    n = 14
    B = 100

    method = "Ridge"

    lmb = 1e-4
    compaire_CV_B(generate_data(N, z_noise, seed=4155), n, B, k_fold_number, method, lamda = lmb)

    lmb = 1e-3
    compaire_CV_B(generate_data(N, z_noise, seed=4155), n, B, k_fold_number, method, lamda = lmb)

    lmb = 1e-2
    compaire_CV_B(generate_data(N, z_noise, seed=4155), n, B, k_fold_number, method, lamda = lmb)

    lmb = 1e2
    compaire_CV_B(generate_data(N, z_noise, seed=4155), n, B, k_fold_number, method, lamda = lmb)
