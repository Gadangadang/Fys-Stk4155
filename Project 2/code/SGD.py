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
        self.lmbd = 0   # Ridge regularization parameter
        self.gamma = 0  # momentum parameter
        self.vel = 0    # gradient descent "velocity"

        # Set gradient and learning rate functions

        if gradient_func == "Logistic":
            self.gradient_func = self.gradient_Logistic

        else:
            self.gradient_func = self.gradient_Ridge # Default

        self.eta_func = self.eta_const # Default

        # Initializa theta and set epoch = 1
        self.reset()


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
        if self.N%self.m != 0:
            batches.append(idx[int_max*self.m:])
        return batches


    #
    # def SGD_evolve(self):
    #     """
    #     Evolve gradient with one epoch
    #     """
    #     batches = self.get_batches()
    #     for batch in batches:
    #         X = self.X[batch]
    #         y = self.y[batch]
    #         g = self.gradient_func(X,y)
    #         self.theta -= self.eta_func(self.epoch) * g  # Update theta
    #

    def SGD_evolve(self):
        """
        Evolve theta with one epoch
        """
        batches = self.get_batches()
        for batch in batches:
            X = self.X[batch]
            y = self.y[batch]
            g = self.gradient_func(X,y)
            self.vel = self.gamma*self.vel + self.eta_func(self.epoch) * g
            self.theta -= self.vel

            # self.theta -= self.eta_func(self.epoch) * g  # Update theta


    def SGD_train(self):
        """
        Run all epochs of SGD
        """
        while self.epoch <= self.num_epochs:
            self.SGD_evolve()
            self.epoch +=1
        return self.theta.ravel()

    def predict(self, X):
        prediction = np.exp(X @ self.theta)/(1+np.exp(X @ self.theta))
        return prediction

    def accuracy_score(self, X, target):

        val = np.sum(np.around(self.predict(X)) == target) / len(target)
        return val

    # --- eta and gradient functions --- #

    def eta_const(self, epoch): # Constant learning rate
        return self.eta_val

    def gradient_Logistic(self, X, y): # Logistic gradient

        g = -X.T @ (y - np.exp(X @ self.theta)/(1+np.exp(X @ self.theta)))
        return g

    def gradient_Ridge(self, X, y): # Ridge gradient
        g = 2*(1/self.N * X.T @ ((X @ self.theta) - y) + self.lmbd*self.theta)
        return g



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
    solver = SGD(X, z, eta_val=0.1, m = 10, num_epochs = int(1e2))
    solver.gamma = 0.8
    theta_SGD = solver.SGD_train()      # Stochastic Gradient Descent
    theta_OLS = OLS_regression(X, z)  # OLS regression


    ztilde_SGD = (X @ theta_SGD).ravel()
    ztilde_OLS = (X @ theta_OLS).ravel()

    print("SGD: ", MSE(ztilde_SGD, z))
    print("OLS: ", MSE(ztilde_OLS, z))
