import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from Functions import *
from matplotlib.ticker import MaxNLocator
from plot_set import *  # Specifies plotting settings



def bias_variance_tradeoff(N, z_noise, n, B, plot=True):
    """
    write info
    """
<<<<<<< HEAD
    x, y, z = generate_data(N, z_noise, seed=None)
    bias = np.zeros(n+1)
    variance = np.zeros(n+1)
    error = np.zeros(n+1)

    # Print process
    info_string = "Bias-variance analysis, #n: "
    print(f"\r{info_string}0/{n}", end = "")


    for i in range(0,n+1): #For increasing complexity
        print(f"\r{info_string}{i}/{n}", end = "")

        X = create_X(x, y, i)
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
        X_train, X_test = scale_design_matrix(X_train, X_test)
        z_pred = bootstrap(X_train, X_test, z_train, z_test, B)
        bias[i] = np.mean((z_test - np.mean(z_pred, axis = 1, keepdims = True))**2) # axis = 1 => columns
        variance[i] = np.mean(np.var(z_pred, axis = 1))
        error[i] = np.mean(np.mean( (z_test-z_pred)**2, axis = 1, keepdims = True  ))
    print(" (done)")
    n_arr = np.linspace(0,n,n+1)
=======
    x, y, z = generate_data(N, z_noise, seed=2018)
    bias = np.zeros(n + 1)
    variance = np.zeros(n + 1)
    error = np.zeros(n + 1)

    for i in range(0, n + 1):  # For increasing complexity

        X = create_X(x, y, i)

        X_train, X_test, z_train, z_test = train_test_split(
            X, z, test_size=0.2)

        X_train, X_test = scale_design_matrix(X_train, X_test)

        z_pred = bootstrap(X_train, X_test, z_train, z_test, B, "OLS")
        # axis = 1 => columns
        bias[i] = np.mean((z_test - np.mean(z_pred, axis=1, keepdims=True))**2)
        variance[i] = np.mean(np.var(z_pred, axis=1))
        error[i] = np.mean(
            np.mean((z_test - z_pred)**2, axis=1, keepdims=True))

    n_arr = np.linspace(0, n, n + 1)
>>>>>>> main
    if plot:
        plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
        plt.plot(n_arr, bias, label="Bias")
        plt.plot(n_arr, variance, label="Variance")
        plt.plot(n_arr, error, "--", label="Error")

        ax = plt.gca()
        # Force integer ticks on x-axis
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xlabel(r"$n$", fontsize=14)
        # plt.ylabel(r"MSE", fontsize=14)
        plt.legend(fontsize=13)
        plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
        # plt.savefig(f"../article/figures/bias_variance_tradeoff.pdf", bbox_inches="tight")
        plt.show()

    return bias, variance, error


if __name__ == "__main__":
    #--- settings ---#
    N =  22          # Number of points in each dimension
    z_noise = 0.2     # Added noise to the z-value
    n = 14                 # Highest order of polynomial for X
    B = 100

    bias_variance_tradeoff(N, z_noise, n, B, plot=True)
