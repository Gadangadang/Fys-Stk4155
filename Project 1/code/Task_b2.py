import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from Functions import *
from matplotlib.ticker import MaxNLocator
from sklearn.utils import resample
from plot_set import * # Specifies plotting settings


# def bias(z, ztilde):
#     return np.mean( (z - np.mean(ztilde))**2 )
#
# def variance(ztilde):
#     return np.mean( (ztilde - np.mean(ztilde))**2 )

def bootstrap(X_train, X_test, z_train, z_test, B):
    """
    info
    """
    z_pred = np.zeros((z_test, B))

    for i in range(B):
        beta_OLS = OLS_regression(X_train, z_train)
        z_pred[:,i] = X_test @ beta_OLS

    return z_pred





def bias_variance_tradeoff(N, z_noise, n, B, plot = True):



    for i in range(0,n+1):
        X = create_X(x, y, i)
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # Force first column of X back to 1
        X_train[:,0] = 1.
        X_test[:,0] = 1.

        beta_OLS = OLS_regression(X_train, z_train)

        ztilde = X_train @ beta_OLS
        zpredict = X_test @ beta_OLS

        bias_array[i] = bias(z_train, ztilde)
        variance_array[i] = variance(ztilde)
        n_arr = np.linspace(0,n,n+1)



    # bias_array, variance_array = np.zeros(n+1), np.zeros(n+1)
    # x, y, z = generate_data(N, z_noise)
    # z = standard_scale(z)
    #
    #
    # for i in range(0,n+1):
    #     X = create_X(x, y, i)
    #     X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
    #
    #     scaler = StandardScaler()
    #     scaler.fit(X_train)
    #     X_train = scaler.transform(X_train)
    #     X_test = scaler.transform(X_test)
    #
    #     # Force first column of X back to 1
    #     X_train[:,0] = 1.
    #     X_test[:,0] = 1.
    #
    #     beta_OLS = OLS_regression(X_train, z_train)
    #
    #     ztilde = X_train @ beta_OLS
    #     zpredict = X_test @ beta_OLS
    #
    #     bias_array[i] = bias(z_train, ztilde)
    #     variance_array[i] = variance(ztilde)
    #     n_arr = np.linspace(0,n,n+1)



    if plot:
        plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
        plt.plot(n_arr, bias_array, label = "Bias")
        plt.plot(n_arr, variance_array, label = "Variance")
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
    N = 50             # Number of points in each dimension
    z_noise = 0.5     # Added noise to the z-value
    n = 20                 # Highest order of polynomial for X


    # bias_variance_tradeoff(N, z_noise, n)
