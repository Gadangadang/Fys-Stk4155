import numpy as np
import matplotlib.pyplot as plt
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from Functions import *
from matplotlib.ticker import MaxNLocator

# from plot_set import * # Specifies plotting settings
from tqdm import trange
from prediction_plots import show_predictions


def Error_Complexity(N, z_noise, n, plot=True, seed=4155):
    MSE_test, MSE_train = np.zeros(n + 1), np.zeros(n + 1)
    x, y, z = generate_data(N, z_noise, seed)


    for i in range(0, n + 1):
        X = create_X(x, y, i)

        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
        mean_scale(X_train, X_test, z_train, z_test)


        beta_OLS = OLS_regression(X_train, z_train)

        ztilde = (X_train @ beta_OLS).ravel()
        zpredict = (X_test @ beta_OLS).ravel()

        MSE_train[i] = MSE(z_train, ztilde)
        MSE_test[i] = MSE(z_test, zpredict)


    if plot:
        #---Plotting---#
        n_arr = np.linspace(0,n,n+1)

        plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
        plt.plot(n_arr[1:], MSE_train[1:], label = "Train")
        plt.plot(n_arr[1:], MSE_test[1:], label = "Test")
        plt.xlabel(r"$n$", fontsize=14)
        plt.ylabel(r"MSE", fontsize=14)
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # Force integer ticks on x-axis
        plt.legend(fontsize = 13)
        plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
        plt.savefig("../article/figures/hastie_copycat.pdf", bbox_inches="tight")

        plt.show()

    return MSE_test, MSE_train

def complexity_bootstrap(N, z_noise, n, B, plot = True, seed = 4155):
    """
    Did not turn out that great...
    """
    MSE_test, MSE_train = np.zeros(n + 1), np.zeros(n + 1)
    x, y, z = generate_data(N, z_noise, seed)


    for i in trange(0, n + 1):
        X = create_X(x, y, i)

        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
        mean_scale(X_train, X_test, z_train, z_test)

        z_pred, z_tilde = bootstrap(X_train, X_test, z_train, z_test, B, method = "OLS", lamda=0, include_train=True)

        z_pred, z_tilde = np.mean(z_pred, axis = 1), np.mean(z_tilde, axis = 1)

        MSE_train[i] = MSE(z_train, z_tilde)
        MSE_test[i] = MSE(z_test, z_pred)


    if plot:
        #---Plotting---#
        n_arr = np.linspace(0,n,n+1)

        plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
        plt.plot(n_arr[1:], MSE_train[1:], label = "Train")
        plt.plot(n_arr[1:], MSE_test[1:], label = "Test")
        plt.xlabel(r"$n$", fontsize=14)
        plt.ylabel(r"MSE", fontsize=14)
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # Force integer ticks on x-axis
        plt.legend(fontsize = 13)
        plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
        plt.savefig(f"../article/figures/hastie_bootstrap{B}.pdf", bbox_inches="tight")

        plt.show()




def multiple_avg(N, z_noise, n, numRuns):
    error = np.zeros((numRuns, 2, n + 1))  # [i, (error_test, error_train), n]

    # Print process
    info_string = "Multirun avg, #run: "
    # Perform multiple runs
    for i in trange(numRuns):
        error_test, error_train = Error_Complexity(N, z_noise, n, plot = False, seed = None)
        error[i,0] = error_test
        error[i,1] = error_train
    # print(" (done)")

    #Find average
    avg = np.mean(error, axis = 0)
    n_arr = np.linspace(0,n,n+1)

    #---Plotting---#
    plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
    plt.title(f"Avg. MSE for {numRuns} runs with {N} datapoints (noise =  {z_noise}" + r" $\times$ $N(0,1)$)")
    plt.plot(n_arr[1:], avg[0][1:], label = "Test")
    plt.plot(n_arr[1:], avg[1][1:], label = "Train")
    ax = plt.gca()
    # Force integer ticks on x-axis
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel(r"$n$", fontsize=14)
    plt.ylabel(r"MSE", fontsize=14)
    plt.legend(fontsize=13)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    plt.savefig(f"../article/figures/hastie_multi{numRuns}.pdf", bbox_inches="tight")
    plt.show()





if __name__ == "__main__":
    N = 25              # Number of points in each dimension
    z_noise = 0.2       # Added noise to the z-value
    n = 25              # Highest order of polynomial for X
    B = 100

    n = np.linspace(2,40, 40-1).astype("int")
    show_predictions(N, z_noise, n)
    # Error_Complexity(N, z_noise, n, plot = True, seed = 4155)
    # complexity_bootstrap(N, z_noise, n, B = 5000)
    # multiple_avg(N, z_noise, n=15, numRuns = 50) # This is not a great solution (talked to TA)
