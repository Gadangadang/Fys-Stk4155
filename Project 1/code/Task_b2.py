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
    test_size = 0.2

    if type(N) == int:
        N = np.array([N])
    else:
        N = np.array(N)

    if B == "N":
        B = ((1-test_size)*N**2).astype(int)


    Nnum = len(N)
    info_string = "Bias-variance analysis, N = "
    for N_i in range(Nnum):
        print(f"\r{info_string}{N[N_i]}, n = 0/{n}", end = "")
        x, y, z = generate_data(N[N_i], z_noise, seed=4155)
        z = standard_scale(z)
        bias = np.zeros(n+1)
        variance = np.zeros(n+1)
        MSE_test = np.zeros(n+1)

        # man_MSE_test = np.zeros(n+1)
        # man_var_test = np.zeros(n+1)
        # man_bias_test = np.zeros(n+1)


        # Print process
        for i in range(0,n+1): #For increasing complexity
            print(f"\r{info_string}{N[N_i]}, n = {i}/{n} ", end = "")

            X = create_X(x, y, i)
            X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=test_size)
            X_train, X_test = scale_design_matrix(X_train, X_test)
            z_pred, z_tilde = bootstrap(X_train, X_test, z_train, z_test, B[N_i], "OLS", lamda=0, include_train=True)

            bias[i] = np.mean((z_test - np.mean(z_pred, axis = 1, keepdims = True))**2) # axis = 1 => columns
            variance[i] = np.mean(np.var(z_pred, axis = 1))
            MSE_test[i] = np.mean(np.mean((z_test-z_pred)**2, axis = 1, keepdims = True  ))


            # Manual testing
            # for j in range(z_pred.shape[1]):
            #     man_MSE_test[i] += MSE(z_test, z_pred[:,j])
            #
            # for k in range(z_pred.shape[0]):
            #     man_var_test[i] += np.mean((z_pred[k,:] - np.mean(z_pred[k,:]))**2)
            #     man_bias_test[i] += np.mean((z_test[k,0] - np.mean(z_pred[k,:]))**2)
            #
            # man_MSE_test[i] /= z_pred.shape[1]
            # man_var_test[i] /= z_pred.shape[0]
            # man_bias_test[i] /= z_pred.shape[0]


        n_arr = np.linspace(0,n,n+1)


        if plot:
            n_arr = np.linspace(0, n, n + 1)

            if Nnum <= 1:
                error_sum = bias + variance
                plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
                plt.title(f"N = {N[N_i]}")
                plt.plot(n_arr[1:], bias[1:], "o-", label=r"Bias$^2$")
                plt.plot(n_arr[1:], variance[1:], "o-", label="Variance")
                plt.plot(n_arr[1:], MSE_test[1:], "o-", label="MSE test")
                plt.fill_between(n_arr[1:], 0, error_sum[1:], alpha = 0.3, color = color_cycle(3), label = r"Bias$^2$ + variance")

                # Manual testing plot
                # plt.plot(n_arr[1:], man_MSE_test[1:], alpha = 0.5, label="man_MSE_test")
                # plt.plot(n_arr[:], man_var_test[:], alpha = 0.5, label="man_var_test")
                # plt.plot(n_arr[:], man_bias_test[:], alpha = 0.5, label="man_bias_test")

                ax = plt.gca()
                # Force integer ticks on x-axis
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                plt.xlabel(r"$n$", fontsize=14)
                # plt.ylabel(r"MSE", fontsize=14)
                plt.legend(fontsize=13)
                plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
                # plt.savefig(f"../article/figures/bias_variance_tradeoff.pdf", bbox_inches="tight")
            elif Nnum == 4:

                plt.figure(num=0, figsize = (8,6), facecolor='w', edgecolor='k')
                plt.subplot(2,2,N_i+1)
                plt.title(f"N = {N[N_i]}")
                plt.plot(n_arr[1:], bias[1:], "o-", label=r"Bias$^2$")
                plt.plot(n_arr[1:], variance[1:], "o-", label="Variance")
                plt.plot(n_arr[1:], MSE_test[1:], "o-", label="MSE test")

                # Manual testing plot
                # plt.plot(n_arr[1:], man_MSE_test[1:], alpha = 0.5, label="man_MSE_test")
                # plt.plot(n_arr[:], man_var_test[:], alpha = 0.5, label="man_var_test")
                # plt.plot(n_arr[:], man_bias_test[:], alpha = 0.5, label="man_bias_test")

                ax = plt.gca()
                # Force integer ticks on x-axis
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                plt.xlabel(r"$n$", fontsize=14)
                # plt.ylabel(r"MSE", fontsize=14)
                if N_i == 0:
                    plt.legend(fontsize=13)
                plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
                if N_i == 4:
                    pass
                    # plt.savefig(f"../article/figures/bias_variance_tradeoff_2x2.pdf", bbox_inches="tight")
    print(" (done)")
    plt.show()

    return 0


if __name__ == "__main__":
    #--- settings ---#
    N =  22          # Number of points in each dimension
    N = [20, 24, 30, 40]
    N = [10, 15, 20, 25]
    z_noise = 0.2     # Added noise to the z-value
    n = 15                # Highest order of polynomial for X
    B = "N"             # Number of training points

    bias_variance_tradeoff(N, z_noise, n, B, plot=True)
