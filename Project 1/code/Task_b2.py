import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from Functions import *
from matplotlib.ticker import MaxNLocator
from plot_set import *  # Specifies plotting settings


def bias_variance_tradeoff(N, z_noise, n, B, method, lamda=[0], plot=True):
    """Calculates the bias variance tradeoff with a given
       linear regression model. Then generates a plot showing
       the tradeoff.

    Args:
        N                (Int): Dimension on the datasets
        z_noise        (Float): Noise scalar, scales the normally distributed noise
        n                (Int): Highest order complexity
        B                (Int): Amount of bootstrap iterations
        method        (String): Choice of linear regression model
        lamda (list, optional): List of floats with adjustment parameter lamda. Defaults to [0].
        plot  (bool, optional): Choice to produce plot or not. Defaults to True.

    Returns:
        Int: Returns zero
    """
    test_size = 0.2
    if type(N) == int:
        N = np.array([N])
    else:
        N = np.array(N)

    if B == "N":
        B = ((1 - test_size) * N**2).astype(int)

    lamdaNum = len(lamda)
    Nnum = len(N)
    info_string = "Bias-variance analysis, N = "
    k = 0
    for lmb in lamda:
        for N_i in range(Nnum):
            print(f"\r{info_string}{N[N_i]}, n = 0/{n}", end="")
            x, y, z = generate_data(N[N_i], z_noise, seed=4155)

            bias = np.zeros(n + 1)
            variance = np.zeros(n + 1)
            MSE_test = np.zeros(n + 1)

            # Print process
            for i in range(0, n + 1):  # For increasing complexity
                print(f"\r{info_string}{N[N_i]}, n = {i}/{n} ", end="")

                X = create_X(x, y, i)
                X_train, X_test, z_train, z_test = train_test_split(
                    X, z, test_size=test_size)
                mean_scale(X_train, X_test, z_train, z_test)

                z_pred, z_tilde = bootstrap(
                    X_train, X_test, z_train, z_test, B[N_i], method, lmb, include_train=True)

                # axis = 1 => columns
                bias[i] = np.mean(
                    (z_test - np.mean(z_pred, axis=1, keepdims=True))**2) - z_noise**2
                variance[i] = np.mean(np.var(z_pred, axis=1))
                MSE_test[i] = np.mean(
                    np.mean((z_test - z_pred)**2, axis=1, keepdims=True))

            n_arr = np.linspace(0, n, n + 1)

            error_sum = bias + variance + z_noise**2
            if plot:
                n_arr = np.linspace(0, n, n + 1)

                if Nnum <= 1 and lamdaNum <= 1:

                    plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
                    if method == "OLS":
                        plt.title(f"{method}: N = {N[N_i]}")
                    else:
                        plt.title(
                            f"{method} $\lambda = ${lmb:.3f} : N = {N[N_i]}")
                    plt.plot(n_arr[1:], bias[1:], "o-", label=r"Bias$^2$")
                    plt.plot(n_arr[1:], variance[1:], "o-", label="Variance")
                    plt.plot(n_arr[1:], MSE_test[1:], "o-", label="MSE test")
                    plt.fill_between(n_arr[1:], 0, error_sum[1:], alpha=0.3, color=color_cycle(
                        3), label=r"Bias$^2$ + variance + $\sigma^2$")

                    ax = plt.gca()
                    # Force integer ticks on x-axis
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                    plt.xlabel(r"$n$", fontsize=14)
                    # plt.ylabel(r"MSE", fontsize=14)
                    plt.legend(fontsize=13)
                    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
                    plt.savefig(
                        f"../article/figures/bias_variance_tradeoff_{method}.pdf", bbox_inches="tight")
                elif Nnum == 4 or lamdaNum == 4:

                    plt.figure(num=0, figsize=(8, 6),
                               facecolor='w', edgecolor='k')
                    plt.subplot(2, 2, k + 1)
                    if method == "OLS":
                        plt.title(f"{method}: N = {N[N_i]}", size=14)
                    else:
                        plt.title(
                            f"{method} $\lambda = ${lmb:.3f} : N = {N[N_i]}", size=14)
                    plt.plot(n_arr[1:], bias[1:], "o-", label=r"Bias$^2$")
                    plt.plot(n_arr[1:], variance[1:], "o-", label="Variance")
                    plt.plot(n_arr[1:], MSE_test[1:], "o--", label="MSE test")
                    plt.fill_between(n_arr[1:], 0, error_sum[1:], alpha=0.3, color=color_cycle(
                        3), label=r"Bias$^2$ + variance + $\sigma^2$")

                    ax = plt.gca()
                    # Force integer ticks on x-axis
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                    plt.xlabel(r"$n$", fontsize=14)

                    # plt.ylabel(r"MSE", fontsize=14)
                    plt.tight_layout()
                    if k == 0:
                        plt.legend(fontsize=13)
                    if N_i == Nnum - 1:
                        plt.savefig(
                            f"../article/figures/bias_variance_tradeoff_{method}_2x2.pdf", bbox_inches="tight")

                    # if k == Nnum or k == lamdaNum:
                    #
                    #     plt.savefig(f"../article/figures/bias_variance_tradeoff_2x2.pdf", bbox_inches="tight")
            k += 1

    print(" (done)")
    plt.show()

    return 0


if __name__ == "__main__":
    #--- settings ---#
    N = 22          # Number of points in each dimension
    #N = [10, 17, 27, 41]?
    z_noise = 0.2     # Added noise to the z-value
    n = 15                # Highest order of polynomial for X
    B = "N"             # Number of training points
    method = "OLS"

    bias_variance_tradeoff(N, z_noise, n, B, method, plot=True)
