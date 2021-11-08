from SGD import *
import seaborn as sns; sns.set_theme()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import MaxNLocator
from plot_set import*

# from matplotlib.ticker import FormatStrFormatter
# from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

# Get modules from project 1
path = os.getcwd()  # Current working directory
path += '/../../Project 1/code'
sys.path.append(path)
from Functions import *

def standard_scale(*args):
    scaled = []
    for arg in args:
        scaler = StandardScaler()
        scaler.fit(arg)
        scaled.append(scaler.transform(arg))

    if len(args) == 1:  # If just one argument
        return scaled[0]
    else:
        return scaled

def mean_scale_new(*args):
    scaled = []
    for arg in args:
        arg =  arg - np.mean(arg, axis=0)
        scaled.append(arg)

    if len(args) == 1:  # If just one argument
        return scaled[0]
    else:
        return scaled



def SGD_optimization_test(X, y):
    """
    Test convergence for different learning rates
    and mini badges
    """
    # np.random.seed(4155) #RN Seed

    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_test, y_train, y_test = mean_scale_new(X_train, X_test, y_train, y_test)
    N = X_train.shape[0]


    # eta_vals = np.logspace(-6, -1, 6)
    eta_vals = np.array([0.001, 0.01, 0.05, 0.1])
    eta_vals = np.array([0.001, 0.2])

    bsp_vals = np.array([0.001, 0.01, 0.5, 1])


    num_epochs = int(1e2)


    SGD_regression = SGD(X_train, y_train, eta_vals[0], m=0, num_epochs = num_epochs)


    train_MSE = np.zeros((len(eta_vals), len(bsp_vals)))
    test_MSE = np.zeros((len(eta_vals), len(bsp_vals)))
    for i, eta_val in enumerate(eta_vals):
        for j, bsp_pct in enumerate(bsp_vals):
            print(f"\r({i},{j})/({len(eta_vals)-1},{len(bsp_vals)-1})", end = "")

            m = int(bsp_pct*N)


            # Find theta
            np.random.seed(4155) #RN Seed
            SGD_regression.reset()
            SGD_regression.eta_val = eta_val
            SGD_regression.m = m
            theta_SGD = SGD_regression.SGD_train()      # Stochastic Gradient Descent


            # Make prediction
            ztilde_theta = (X_train @ theta_SGD).ravel()
            zpredict_theta = (X_test @ theta_SGD).ravel()


            # Error
            train_MSE[i,j] = MSE(ztilde_theta, y_train)
            test_MSE[i,j] = MSE(zpredict_theta, y_test)



    print()


    fig, ax = plt.subplots(figsize=(7, 7))
    # df = pd.DataFram(test_MSE), columns
    ax = sns.heatmap(test_MSE, xticklabels = bsp_vals, yticklabels = np.log10(eta_vals), annot=True, ax=ax, cmap="viridis")
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax.set_title("Test MSE")
    ax.set_ylabel("$\log{\eta}$")
    ax.set_xlabel("bsp_pct")
    plt.show()


def SGD_test_learning_rate(X, y):
    """
    Full gradient descent
    for different learning rates
    """


    # Split and scale data
    np.random.seed(415895) # To avoid unfortunately split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_test, y_train, y_test = mean_scale_new(X_train, X_test, y_train, y_test)
    N = X_train.shape[0]

    # eta_vals = np.logspace(-6, -2, 5)
    eta_vals = np.linspace(0.001, 1.5875, 100)


    num_epochs = int(1e3)
    m = 0

    SGD_regression = SGD(X_train, y_train, eta_vals[0], m=m, num_epochs = num_epochs)


    SGD_MSE_train = np.zeros(len(eta_vals))
    SGD_MSE_test = np.zeros(len(eta_vals))
    for i, eta_val in enumerate(eta_vals):
        print(f"\r{i}/{len(eta_vals)-1}", end = "")

        # Find theta
        np.random.seed(4155) #RN Seed
        SGD_regression.reset()
        SGD_regression.eta_val = eta_val
        theta_SGD = SGD_regression.SGD_train()

        # Make prediction
        ztilde_theta = (X_train @ theta_SGD).ravel()
        zpredict_theta = (X_test @ theta_SGD).ravel()


        # Error
        SGD_MSE_train[i] = MSE(ztilde_theta, y_train)
        SGD_MSE_test[i] = MSE(zpredict_theta, y_test)
    print()

    # OLS regression
    theta_OLS = OLS_regression(X_train, y_train)
    ztilde = (X_train @ theta_OLS).ravel()
    zpredict = (X_test @ theta_OLS).ravel()

    # MSE for OLS
    OLS_MSE_train = np.ones(len(eta_vals)) * MSE(ztilde, y_train)
    OLS_MSE_test = np.ones(len(eta_vals)) * MSE(zpredict, y_test)
    print("test min arg:", eta_vals[np.argmin(SGD_MSE_test)], "+-", eta_vals[1]-eta_vals[0])
    print("argmin idx:", np.argmin(SGD_MSE_test))

    #--- Plotting ---#
    plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
    plt.plot(eta_vals, SGD_MSE_train, label = "SGD train")
    plt.plot(eta_vals, SGD_MSE_test, "--", label = "SGD test")
    plt.plot(eta_vals, OLS_MSE_train, label = "OLS train")
    plt.plot(eta_vals, OLS_MSE_test, "--", label = "OLS test")

    ax = plt.gca()
    plt.yscale("log")
    ax.yaxis.grid(True, which='minor')

    plt.xlabel(r"$\eta$", fontsize=14)
    plt.ylabel(r"MSE train", fontsize=14)
    plt.legend(fontsize = 13)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    plt.savefig("../article/figures/SGD_learning_rate_MSE.pdf", bbox_inches="tight")
    plt.show()


def SGD_convergence_rate(X, y):
    N = X.shape[0]

    #--- Variation of batch size, constant learning rate ---#

    # settings
    num_epochs = 10
    eta_val = 0.1
    bsp_vals = np.array([0.0001, 0.001, 0.01, 0.5, 1])


    SGD_regression = SGD(X, y, eta_val, m=0, num_epochs = num_epochs)


    theta_OLS = OLS_regression(X, y)
    ytilde_OLS = (X @ theta_OLS).ravel()
    MSE_OLS =  MSE(ytilde_OLS, y)


    fig = plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
    ax = plt.gca()
    MSE_arr = np.zeros((len(bsp_vals), num_epochs+1))
    epoch_arr = np.linspace(0, num_epochs, num_epochs+1)
    for i, bsp_pct in enumerate(bsp_vals):
        np.random.seed(4155) #RN Seed
        SGD_regression.reset()
        SGD_regression.m = int(bsp_pct*N)

        ytilde = (X @ SGD_regression.theta).ravel()
        MSE_arr[i,0] = MSE(ytilde, y)

        for j in range(1,num_epochs+1):
            SGD_regression.SGD_evolve()
            ytilde = (X @ SGD_regression.theta).ravel()
            MSE_arr[i,j] = MSE(ytilde, y)
        plt.plot(epoch_arr, MSE_arr[i], "-o", markersize = 4, label = f"m = {SGD_regression.m} ({bsp_pct*100:g}%)")


    plt.xlabel(r"epoch", fontsize=14)
    plt.ylabel(r"MSE train", fontsize=14)
    plt.yscale("log")
    ax.yaxis.grid(True, which='minor')
    plt.legend(fontsize = 13)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    plt.savefig("../article/figures/SGD_batch_size_convergence.pdf", bbox_inches="tight")


    #--- Variation of learning rate, constant batch size ---#

    # settings
    num_epochs = 500
    eta_vals = np.array([0.001, 0.01, 0.05, 0.1])

    SGD_regression = SGD(X, y, eta_val, m=0, num_epochs = num_epochs)



    fig = plt.figure(num=1, dpi=80, facecolor='w', edgecolor='k')
    ax = plt.gca()
    plt.hlines(MSE_OLS, 1, num_epochs, linestyle = "--", color = "black", label = "OLS MSE")

    MSE_arr = np.zeros((len(bsp_vals), num_epochs+1))
    epoch_arr = np.linspace(0, num_epochs, num_epochs+1)
    for i, eta_val in enumerate(eta_vals):
        np.random.seed(4155) #RN Seed
        SGD_regression.reset()
        SGD_regression.eta_val = eta_val

        ytilde = (X @ SGD_regression.theta).ravel()
        MSE_arr[i,0] = MSE(ytilde, y)

        for j in range(1,num_epochs+1):
            SGD_regression.SGD_evolve()
            ytilde = (X @ SGD_regression.theta).ravel()
            MSE_arr[i,j] = MSE(ytilde, y)
        plt.plot(epoch_arr, MSE_arr[i], label = "$\eta$" + f" = {eta_val:g}")



    plt.xlabel(r"epoch", fontsize=14)
    plt.ylabel(r"MSE train", fontsize=14)
    plt.yscale("log")
    ax.yaxis.grid(True, which='minor')
    plt.legend(fontsize = 13)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    plt.savefig("../article/figures/SGD_learning_rate_convergence.pdf", bbox_inches="tight")
    plt.show()



def SGD_VS_OLS():
    """
    SGD VS OLS
    """

    #--- Create data from Franke Function ---#
    N = 20             # Number of points in each dimension
    z_noise = 0.2      # Added noise to the z-value
    n_max = 20
    x, y, z = generate_data(N, z_noise)

    split = 0.2
    num_epochs = int(1e4)
    X_full = create_X(x, y, n_max)
    X_F_train, X_F_test, y_train, y_test = train_test_split(X_full, z, test_size=split)
    X_F_train, X_F_test, y_train, y_test = mean_scale_new(X_F_train, X_F_test, y_train, y_test)


    eta_val = 0.1
    m = 0




    MSE_train_SGD = np.zeros(n_max)
    MSE_test_SGD = np.zeros(n_max)
    MSE_train_OLS = np.zeros(n_max)
    MSE_test_OLS = np.zeros(n_max)

    n_arr = np.arange(1,n_max+1)
    for i, n in enumerate(n_arr):
        print(f"\rn: {n}/{n_max}", end = "")
        l = int((n+1)*(n+2)/2)

        X_train = X_F_train[:,0:l]
        X_test = X_F_test[:,0:l]

        # SGD
        np.random.seed(4155) #RN Seed
        SGD_regression = SGD(X_train, y_train, eta_val, m=m, num_epochs = num_epochs)
        theta_SGD = SGD_regression.SGD_train()

        ztilde_theta = (X_train @ theta_SGD).ravel()
        zpredict_theta = (X_test @ theta_SGD).ravel()
        MSE_train_SGD[i] = MSE(ztilde_theta, y_train)
        MSE_test_SGD[i] = MSE(zpredict_theta, y_test)


        # OLS
        theta_OLS = OLS_regression(X_train, y_train)
        ztilde = (X_train @ theta_OLS).ravel()
        zpredict = (X_test @ theta_OLS).ravel()
        MSE_train_OLS[i] = MSE(ztilde, y_train)
        MSE_test_OLS[i] = MSE(zpredict, y_test)
    print()


    plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
    ax = plt.gca()
    plt.plot(n_arr, MSE_train_SGD, "-o", markersize = 4, label = "SGD train")
    plt.plot(n_arr, MSE_test_SGD, "-o", markersize = 4, label = "SGD test ")
    plt.plot(n_arr, MSE_train_OLS, "-o", markersize = 4, label = "OLS train")
    plt.plot(n_arr, MSE_test_OLS, "-o", markersize = 4, label = "OLS test")

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel(r"$n$", fontsize=14)
    plt.ylabel(r"MSE", fontsize=14)
    plt.legend(fontsize = 13)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    plt.savefig("../article/figures/SGD_VS_OLS.pdf", bbox_inches="tight")
    plt.show()

def SGD_VS_Ridge(X, y):
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_test, y_train, y_test = mean_scale_new(X_train, X_test, y_train, y_test)
    N = X_train.shape[0]

    # eta_vals = np.logspace(-6, -1, 6)
    eta_vals = np.array([0.001, 0.01, 0.05, 0.1])
    lmbd_vals = np.logspace(-8, 0, 9)
    num_epochs = int(1e4)

    SGD_regression = SGD(X_train, y_train, eta_vals[0], m=0, num_epochs = num_epochs)
    train_MSE = np.zeros((len(eta_vals), len(lmbd_vals)))
    test_MSE = np.zeros((len(eta_vals), len(lmbd_vals)))
    for i, eta_val in enumerate(eta_vals):
        for j, lmbd in enumerate(lmbd_vals):
            print(f"\r({i},{j})/({len(eta_vals)-1},{len(lmbd_vals)-1})", end = "")

            # Find theta
            np.random.seed(4155) #RN Seed
            SGD_regression.reset()
            SGD_regression.eta_val = eta_val
            SGD_regression.lmbd = lmbd
            theta_SGD = SGD_regression.SGD_train()      # Stochastic Gradient Descent

            # Make prediction
            ztilde_theta = (X_train @ theta_SGD).ravel()
            zpredict_theta = (X_test @ theta_SGD).ravel()


            # Error
            train_MSE[i,j] = MSE(ztilde_theta, y_train)
            test_MSE[i,j] = MSE(zpredict_theta, y_test)


    print()


    fig, ax = plt.subplots(num = 0, figsize=(7, 7))
    ax = sns.heatmap(test_MSE, xticklabels = np.log10(lmbd_vals), yticklabels = eta_vals, annot=True, ax=ax, cmap="viridis")

    ax.set_title("SGD Ridge")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\log{\ \lambda}$")
    ax.collections[0].colorbar.set_label("MSE test")
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    plt.savefig("../article/figures/SGD_Ridge_heatmap.pdf", bbox_inches="tight")

    exit()

    # SGD VS Ridge
    num_epochs = int(1e4)
    eta_val = 0.1
    m = 0

    SGD_regression = SGD(X_train, y_train, eta_val, m=m, num_epochs = num_epochs)

    MSE_train_SGD = np.zeros(len(lmbd_vals))
    MSE_test_SGD = np.zeros(len(lmbd_vals))
    MSE_train_Ridge = np.zeros(len(lmbd_vals))
    MSE_test_Ridge = np.zeros(len(lmbd_vals))
    for i, lmbd in enumerate(lmbd_vals):
        print(f"\rlmbd: {i}/{len(lmbd_vals)}", end = "")

        # SGD
        np.random.seed(41550) #RN Seed
        SGD_regression.reset()
        SGD_regression.lmbd = lmbd
        theta_SGD = SGD_regression.SGD_train()


        ztilde_theta = (X_train @ theta_SGD).ravel()
        zpredict_theta = (X_test @ theta_SGD).ravel()
        MSE_train_SGD[i] = MSE(ztilde_theta, y_train)
        MSE_test_SGD[i] = MSE(zpredict_theta, y_test)


        # Ridge
        theta_Ridge = RIDGE_regression(X_train, y_train, lmbd)
        ztilde = (X_train @ theta_Ridge).ravel()
        zpredict = (X_test @ theta_Ridge).ravel()
        MSE_train_Ridge[i] = MSE(ztilde, y_train)
        MSE_test_Ridge[i] = MSE(zpredict, y_test)
    print()


    plt.figure(num=1, dpi=80, facecolor='w', edgecolor='k')
    ax = plt.gca()
    # plt.plot(lmbd_vals, MSE_train_SGD, "-o", markersize = 4, label = "SGD train")
    plt.plot(lmbd_vals, MSE_test_SGD, "-o", markersize = 4, label = "SGD test ")
    # plt.plot(lmbd_vals, MSE_train_Ridge, "-o", markersize = 4, label = "Ridge train")
    plt.plot(lmbd_vals, MSE_test_Ridge, "-o", markersize = 4, label = "Ridge test")

    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xscale("log")
    plt.xlabel(r"$\lambda$", fontsize=14)
    plt.ylabel(r"MSE", fontsize=14)
    plt.legend(fontsize = 13)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    plt.savefig("../article/figures/SGD_VS_Ridge.pdf", bbox_inches="tight")
    plt.show()

def SGD_timing_batch_size():
    from timeit import default_timer as timer


    N_arr = np.linspace(100, 1000, 10).astype("int")
    z_noise = 0.2      # Added noise to the z-value
    n = 10

    eta_val = 0.1
    num_epochs = int(10)
    bsp_vals = np.array([0.001, 0.1, 1])

    time = np.zeros((len(bsp_vals), len(N_arr)))


    for j, N_side in enumerate(N_arr):
        #--- Create data from Franke Function ---#
        x, y, z = generate_data(N_side, z_noise)
        X = create_X(x, y, n)


        SGD_regression = SGD(X, z, eta_val = 0.1, m=0, num_epochs = num_epochs)
        for i, bsp_pct in enumerate(bsp_vals):
            print(f"\r N = {N_side}/{N_arr[-1]}  ", end = "")

            np.random.seed(4155) #RN Seed
            SGD_regression.reset()
            SGD_regression.m = int(bsp_pct*N_side**2)


            # Timing
            start = timer()
            SGD_regression.SGD_train()
            end = timer()

            time[i,j] = end - start
    print()

    time /= num_epochs


    matrix_elem = N_arr*(n+1)*(n+2)/2
    fig = plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
    ax = plt.gca()


    for i, bsp_pct in enumerate(bsp_vals):
        plt.plot(matrix_elem, time[i], "-o", label = f"m = {bsp_pct*100:g}%")



    plt.xlabel(r"Total matrix elements", fontsize=14)
    plt.ylabel(r"Time pr. epoch [s]", fontsize=14)
    # plt.yscale("log")
    # plt.xscale("log")
    # ax.yaxis.grid(True, which='minor')
    plt.legend(fontsize = 13)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    plt.savefig("../article/figures/SGD_timing.pdf", bbox_inches="tight")
    plt.show()


def SGD_momentum_convergence_rate(X, y):
        N = X.shape[0]

        #--- Variation of batch size, constant learning rate ---#

        # settings
        num_epochs = 100
        eta_val = 0.01
        gamma_vals = np.array([0, 0.1, 0.3, 0.5, 0.7, 0.9])
        SGD_regression = SGD(X, y, eta_val, m=0, num_epochs = num_epochs)


        theta_OLS = OLS_regression(X, y)
        ytilde_OLS = (X @ theta_OLS).ravel()
        MSE_OLS =  MSE(ytilde_OLS, y)


        fig = plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
        ax = plt.gca()
        MSE_arr = np.zeros((len(gamma_vals), num_epochs+1))
        epoch_arr = np.linspace(0, num_epochs, num_epochs+1)
        for i, gamma in enumerate(gamma_vals):
            np.random.seed(41550) #RN Seed
            SGD_regression.reset()
            SGD_regression.gamma = gamma

            ytilde = (X @ SGD_regression.theta).ravel()
            MSE_arr[i,0] = MSE(ytilde, y)

            for j in range(1,num_epochs+1):
                SGD_regression.SGD_evolve()
                ytilde = (X @ SGD_regression.theta).ravel()
                MSE_arr[i,j] = MSE(ytilde, y)
            # plt.plot(epoch_arr, MSE_arr[i], "-o", label = "$\gamma$ = " + f"{gamma:g}")
            plt.plot(epoch_arr, MSE_arr[i], label = "$\gamma$ = " + f"{gamma:g}")


        plt.title("$\eta$" + f"= {eta_val:g}")
        plt.xlabel(r"epoch", fontsize=14)
        plt.ylabel(r"MSE train", fontsize=14)
        plt.yscale("log")
        ax.yaxis.grid(True, which='minor')
        plt.legend(fontsize = 13)
        plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
        plt.savefig(f"../article/figures/SGD_momentum_convergence_eta{eta_val}.pdf", bbox_inches="tight")
        plt.show()



def SGD_momentum_optimization(X, y):

    np.random.seed(4155) #RN Seed

    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_test, y_train, y_test = mean_scale_new(X_train, X_test, y_train, y_test)
    N = X_train.shape[0]


    # eta_vals = np.logspace(-6, -1, 6)
    eta_vals = np.array([0.001, 0.01, 0.05, 0.1])
    gamma_vals = np.array([0, 0.1, 0.3, 0.5, 0.7, 0.9])

    num_epochs = int(1e1)


    SGD_regression = SGD(X_train, y_train, eta_vals[0], m=0, num_epochs = num_epochs)


    train_MSE = np.zeros((len(eta_vals), len(gamma_vals)))
    test_MSE = np.zeros((len(eta_vals), len(gamma_vals)))
    for i, eta_val in enumerate(eta_vals):
        for j, gamma in enumerate(gamma_vals):
            print(f"\r({i},{j})/({len(eta_vals)-1},{len(gamma_vals)-1})", end = "")


            # Find theta
            np.random.seed(4155) #RN Seed
            SGD_regression.reset()
            SGD_regression.eta_val = eta_val
            SGD_regression.gamma = gamma
            theta_SGD = SGD_regression.SGD_train()      # Stochastic Gradient Descent


            # Make prediction
            ztilde_theta = (X_train @ theta_SGD).ravel()
            zpredict_theta = (X_test @ theta_SGD).ravel()


            # Error
            train_MSE[i,j] = MSE(ztilde_theta, y_train)
            test_MSE[i,j] = MSE(zpredict_theta, y_test)



    print()


    fig, ax = plt.subplots(figsize=(7, 7))
    # df = pd.DataFram(test_MSE), columns
    ax = sns.heatmap(test_MSE, xticklabels = gamma_vals, yticklabels = eta_vals, annot=True, ax=ax, cmap="viridis")
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax.set_title("Test MSE")
    ax.set_ylabel("$\log{\eta}$")
    ax.set_xlabel("$\gamma$")
    plt.show()

























if __name__ == "__main__":


    #--- Create data from Franke Function ---#
    N = 100             # Number of points in each dimension
    z_noise = 0.2       # Added noise to the z-value
    n = 5               # Highest order of polynomial for X
    x, y, z = generate_data(N, z_noise)
    X = create_X(x, y, n)



    #--- Validation / testing ---#
    # SGD_convergence_rate(X, z)
    # SGD_optimization_test(X, z) # not included in report so far
    SGD_test_learning_rate(X, z)
    # SGD_VS_OLS()
    # SGD_VS_Ridge(X,z)
    # SGD_momentum_convergence_rate(X,z)
    # SGD_momentum_optimization(X,z)


    # SGD_timing_batch_size()
