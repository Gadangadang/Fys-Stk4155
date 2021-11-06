from SGD import *
import seaborn as sns; sns.set_theme()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import FormatStrFormatter

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
    np.random.seed(4155) #RN Seed

    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_test, y_train, y_test = mean_scale_new(X_train, X_test, y_train, y_test)
    N = X_train.shape[0]


    eta_vals = np.logspace(-6, -1, 6)  # eta_vals = np.linspace(0.13, 0.1, 2)
    bsp_vals = np.linspace(0.01,1, 5) # batch size procent
    num_epochs = int(1e4)


    SGD_regression = SGD(X_train, y_train, eta_vals[0], m=0, num_epochs = num_epochs)


    train_MSE = np.zeros((len(eta_vals), len(bsp_vals)))
    test_MSE = np.zeros((len(eta_vals), len(bsp_vals)))
    for i, eta_val in enumerate(eta_vals):
        for j, bsp_pct in enumerate(bsp_vals):
            print(f"\r({i},{j})/({len(eta_vals)-1},{len(bsp_vals)-1})", end = "")
            m = int(bsp_pct*N)

            # Find theta
            SGD_regression.reset()
            SGD_regression.initialize_theta_normal()
            SGD_regression.eta_val = eta_val
            SGD_regression.m = m
            theta_SGD = SGD_regression.SGD_run()      # Stochastic Gradient Descent


            # Make prediction
            ztilde_theta = (X_train @ theta_SGD).ravel()
            zpredict_theta = (X_test @ theta_SGD).ravel()


            # Error
            train_MSE[i,j] = MSE(ztilde_theta, y_train)
            test_MSE[i,j] = MSE(zpredict_theta, y_test)



    print()


    fig, ax = plt.subplots(figsize=(7, 7))
    # df = pd.DataFram(test_MSE), columns
    ax = sns.heatmap(train_MSE, xticklabels = bsp_vals, yticklabels = np.log10(eta_vals), annot=True, ax=ax, cmap="viridis")
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
    np.random.seed(41550) #RN Seed

    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_test, y_train, y_test = mean_scale_new(X_train, X_test, y_train, y_test)
    N = X_train.shape[0]

    # eta_vals = np.logspace(-6, -2, 5)
    eta_vals = np.linspace(0.001, 1, 10)
    num_epochs = int(1e5)


    train_MSE = np.zeros(len(eta_vals))
    test_MSE = np.zeros(len(eta_vals))

    m = int(1*N)
    for i, eta_val in enumerate(eta_vals):
        print(f"\r{i}/{len(eta_vals)-1}", end = "")

        # Find theta
        theta = SGD(X_train, y_train, eta_val, m, num_epochs = num_epochs)

        # Make prediction
        ztilde_theta = (X_train @ theta).ravel()
        zpredict_theta = (X_test @ theta).ravel()


        # Error
        train_MSE[i] = MSE(ztilde_theta, y_train)
        test_MSE[i] = MSE(zpredict_theta, y_test)
    print()
    # OLS regression
    beta_OLS = OLS_regression(X_train, y_train)
    ztilde = (X_train @ beta_OLS).ravel()
    zpredict = (X_test @ beta_OLS).ravel()

    # MSE for OLS
    MSE_train = MSE(ztilde, y_train)
    MSE_test = MSE(zpredict, y_test)

    OLS_MSE_train = np.ones(len(eta_vals)) * MSE_train
    OLS_MSE_test = np.ones(len(eta_vals)) * MSE_test

    plt.plot(eta_vals, train_MSE, label = "train MSE SGD")
    plt.plot(eta_vals, test_MSE, label = "test MSE SGD")
    plt.plot(eta_vals, OLS_MSE_train, label = "train MSE OLS")
    plt.plot(eta_vals, OLS_MSE_test, label = "test MSE OLS")
    plt.legend()
    plt.show()


def SGD_VS_OLS():
    """
    SGD VS OLS
    """

    #--- Create data from Franke Function ---#
    N = 10             # Number of points in each dimension
    z_noise = 0.2       # Added noise to the z-value
    n_max = 10
    x, y, z = generate_data(N, z_noise)

    n_max = 12
    split = 0.2
    num_epochs = int(1e5)
    X_full = create_X(x, y, n_max)
    X_F_train, X_F_test, y_train, y_test = train_test_split(X_full, z, test_size=split)
    X_F_train, X_F_test, y_train, y_test = mean_scale_new(X_F_train, X_F_test, y_train, y_test)

    eta_val = 0.1
    m = int((1-split)*N**2) # Full gradient descent


    MSE_train_SGD = np.zeros(n_max)
    MSE_test_SGD = np.zeros(n_max)
    MSE_train_OLS = np.zeros(n_max)
    MSE_test_OLS = np.zeros(n_max)
    # came to this point
    n_arr = np.arange(1,n_max+1)
    for i, n in enumerate(n_arr):
        print(f"\rn: {n}/{n_max}", end = "")
        l = int((n+1)*(n+2)/2)

        X_train = X_F_train[:,0:l]
        X_test = X_F_test[:,0:l]


        theta_SGD = SGD(X_train, y_train, eta_val, m, num_epochs = num_epochs)
        ztilde_theta = (X_train @ theta_SGD).ravel()
        zpredict_theta = (X_test @ theta_SGD).ravel()

        MSE_train_SGD[i] = MSE(ztilde_theta, y_train)
        MSE_test_SGD[i] = MSE(zpredict_theta, y_test)


        beta_OLS = OLS_regression(X_train, y_train)
        ztilde = (X_train @ beta_OLS).ravel()
        zpredict = (X_test @ beta_OLS).ravel()


        MSE_train_OLS[i] = MSE(ztilde, y_train)
        MSE_test_OLS[i] = MSE(zpredict, y_test)



    plt.plot(n_arr, MSE_train_SGD, label = "train MSE SGD")
    plt.plot(n_arr, MSE_test_SGD, label = "test MSE SGD")
    plt.plot(n_arr, MSE_train_OLS, label = "train MSE OLS")
    plt.plot(n_arr, MSE_test_OLS, label = "test MSE OLS")
    plt.legend()
    plt.show()


# def SGD_convergence_speed():
#     #--- Create data from Franke Function ---#
#     N = 10             # Number of points in each dimension
#     z_noise = 0.2       # Added noise to the z-value
#     n_max = 10
#     x, y, z = generate_data(N, z_noise)
#
#     n_max = 12
#
#
















if __name__ == "__main__":


    #--- Create data from Franke Function ---#
    N = 10             # Number of points in each dimension
    z_noise = 0.2       # Added noise to the z-value
    n = 5               # Highest order of polynomial for X
    lamda = 0
    x, y, z = generate_data(N, z_noise)
    X = create_X(x, y, n)



    #--- Validation / testing ---#
    SGD_optimization_test(X, z)
    # SGD_test_learning_rate(X, z)
    # SGD_VS_OLS()
