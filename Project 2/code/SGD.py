import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from tqdm import trange
import matplotlib.pyplot as plt

class SGD:
    """
    Stochastic Gradient Descent
    with mini batches
    """
    def __init__(self, X, y, eta_val=0.1, m = 0, num_epochs = int(1e4),  gradient_func = "Ridge"):
        self.X = X
        self.N = X.shape[0] # Number of data points
        self.y = y

        dim_check = X.shape[0] == y.shape[0]
        assert dim_check, "Dimensions of X and y does not match"

        self.eta_val = eta_val
        if m == 0: # full gradient descent
            self.m = self.N
        else: # user defined m
            self.m = m
        self.num_epochs = num_epochs
        self.lmbd = 0

        # Set default gradient and learning rate functions

        if gradient_func == "logistic":
            self.gradient_func = self.gradient_Logistic

        else:
            self.gradient_func = self.gradient_Ridge

        self.eta_func = self.eta_const

        # Initializa theta and set epoch = 1
        self.reset()
        self.get_batches()


    def reset(self):
        self.epoch = 1
        self.initialize_theta_normal()


    def initialize_theta_normal(self):
        """
        Initialize theta with normal disttribution
        """
        self.theta = np.random.normal(0, 1, size=(self.X.shape[1], 1))


    def get_batches(self):
        idx = np.arange(self.N)
        np.random.shuffle(idx)
        int_max = self.N//self.m
        batches = [idx[i*self.m:(i+1)*self.m] for i in range(int_max)]
        batches.append(idx[int_max*self.m:])
        return batches



    def SGD_evolve(self):
        """
        Evolve gradient with one step
        """
        batch = np.random.choice(self.N, self.m, replace=False)
        X = self.X[batch]
        y = self.y[batch]
        g = self.gradient_func(X,y)
        self.theta -= self.eta_func(self.epoch) * g  # Update theta


    def SGD_train(self):
        """
        Run all epochs of SGD
        """
        while self.epoch <= self.num_epochs:
            self.SGD_evolve()
            self.epoch +=1

        return self.theta.ravel()


    # --- eta and gradient functions --- #

    def eta_const(self, epoch): # Constant learning rate
        return self.eta_val

    def gradient_Logistic(self, X, y): # Ridge gradient
        g = -X.T @ (y - np.exp(X @ self.theta)/(1+np.exp(X @ self.theta)))
        return g

    def gradient_Ridge(self, X, y): # Ridge gradient
        g = 2*(1/self.N * X.T @ ((X @ self.theta) - y) + self.lmbd*self.theta)
        return g






# --- OLD shit soon to be removed --- #


# def SGD(X, y, eta_val=0.1, m = 0, num_epochs = int(1e4)):
#
#
#
#
#     N = X.shape[0]                      # Number of data points
#     if m == 0: # full gradient descent
#         m = N                              # Size of each mini-batch
#                     # Number of epochs
#
#     theta = np.random.normal(0, 1, size=(X.shape[1], 1)) # Initialize theta with normal dist.
#     def eta(epoch): return eta_val            # Learning rate (constant here)
#     for epoch in range(1, num_epochs + 1):
#         batch = np.random.choice(N, m, replace=False)  # Mini batch
#         g = 2.0 / len(y) * X.T @ ((X @ theta) - y)  # Compute gradient
#         #print(g, theta)
#         theta -= eta(epoch) * g  # Update theta
#
#     return theta.ravel()





# def compare_SGD_OLS(X, z, eta):
#     X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
#
#     # Scale bu subtracting mean
#     mean_scale(X_train, X_test, z_train, z_test)
#
#     # OLS regression
#     beta_OLS = OLS_regression(X_train, z_train)
#     ztilde = (X_train @ beta_OLS).ravel()
#     zpredict = (X_test @ beta_OLS).ravel()
#
#     # MSE for OLS
#     MSE_train = MSE(z_train, ztilde)
#     MSE_test = MSE(z_test, zpredict)
#     print("--- OLS ---")
#     print("train err {:.7f} test err {:.7f}".format(MSE_train, MSE_test))
#
#     OLS_MSE_train = np.ones(len(eta)) * MSE_train
#     OLS_MSE_test = np.ones(len(eta)) * MSE_test
#
#     # MSE for SGD
#     SGD_MSE_train = np.zeros(len(eta))
#     SGD_MSE_test = np.zeros(len(eta))
#
#     print("--- SGD ---")
#     for index, eta_val in enumerate(eta):
#         # Find theta
#         theta_train = SGD(X_train, z_train, eta_val=eta_val)
#         theta_test = SGD(X_test, z_test, eta_val=eta_val) <----- This is wrong!
#
#         # Prediction
#         ztilde_theta = (X_train @ theta_train).ravel()
#         zpredict_theta = (X_test @ theta_test).ravel()
#
#         # Error
#         train_err = MSE(ztilde_theta, z_train)
#         test_err = MSE(zpredict_theta, z_test)
#
#         print("train err {:.7f} test err {:.7f}".format(train_err, test_err))
#
#         SGD_MSE_train[index] = train_err
#         SGD_MSE_test[index] = test_err
#
#     plotting(eta, OLS_MSE_test, SGD_MSE_test)
#     #plotting(eta, OLS_MSE_test, SGD_MSE_test)
#
#
# def plotting(eta, OLS_MSE, SGD_MSE):
#     plt.plot(eta, OLS_MSE, "--", label="OLS")
#     plt.plot(eta, SGD_MSE, label="SGD")
#     plt.xlabel(r"$\eta - Learning rate$")
#     plt.ylabel("MSE")
#     plt.legend()
#     plt.show()


if __name__ == "__main__":
    # Get modules from project 1
    path = os.getcwd()  # Current working directory
    path += '/../../Project 1/code'
    sys.path.append(path)
    from Functions import *

    #--- Create data from Franke Function ---#
    N = 10               # Number of points in each dimension
    z_noise = 0.2       # Added noise to the z-value
    n = 3               # Highest order of polynomial for X
    x, y, z = generate_data(N, z_noise)
    X = create_X(x, y, n)


    # --- Test run --- #
    eta = 0.001
    m = 0 # m=0 gives full gradient descent

    #--- Regression ---#
    solver = SGD(X, z, eta_val=0.1, m = 7, num_epochs = int(1e5))
    theta_SGD = solver.SGD_train()      # Stochastic Gradient Descent
    theta_OLS = OLS_regression(X, z)  # OLS regression


    ztilde_SGD = (X @ theta_SGD).ravel()
    ztilde_OLS = (X @ theta_OLS).ravel()

    print("SGD: ", MSE(ztilde_SGD, z))
    print("OLS: ", MSE(ztilde_OLS, z))
