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

    x, y, z = generate_data(N, z_noise, seed=4155)
    bias = np.zeros(n+1)
    variance = np.zeros(n+1)
    error = np.zeros(n+1)
    error_test = np.zeros(n+1)


    # Print process
    info_string = "Bias-variance analysis, #n: "
    print(f"\r{info_string}0/{n}", end = "")


    for i in range(0,n+1): #For increasing complexity
        print(f"\r{info_string}{i}/{n}", end = "")
        X = create_X(x, y, i)
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
        # z_pred = bootstrap(X_train, X_test, z_train, z_test, B, "OLS")
        z_pred, z_tilde = bootstrap(X_train, X_test, z_train, z_test, B, "OLS", lamda=0, include_train=True)
        bias[i] = np.sqrt(np.mean((z_test - np.mean(z_pred, axis = 1, keepdims = True))**2)) # axis = 1 => columns
        variance[i] = np.mean(np.var(z_pred, axis = 1))
        error[i] = np.mean(np.mean( (z_test-z_pred)**2, axis = 1, keepdims = True  ))

        for j in range(z_pred.shape[0]):
            error_test[i] += MSE(z_test, z_pred[j,0])
        error_test[i]/z_pred.shape[0]
    print(" (done)")
    n_arr = np.linspace(0,n,n+1)
    ###### Look into MSE ########


    n_arr = np.linspace(0, n, n + 1)

    if plot:
        plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
        plt.plot(n_arr[1:], bias[1:], label="Bias")
        plt.plot(n_arr[1:], variance[1:], label="Variance")
        plt.plot(n_arr[1:], error[1:], "--", label="MSE test")
        plt.plot(n_arr[1:], error_test[1:], "--", label="Error test 2")


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
    n = 15                 # Highest order of polynomial for X
    B = int(0.8*N**2) # Number of training points

    bias_variance_tradeoff(N, z_noise, n, B, plot=True)
