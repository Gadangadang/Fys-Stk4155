import numpy as np
import matplotlib.pyplot as plt
import os, sys
from sklearn.model_selection import train_test_split
from tqdm import trange

def SGD(X, y):
    """
    Stochastic Gradient Descent
    with mini batches
    """

    dim_check = X.shape[0] == y.shape[0]
    assert dim_check, "Dimensions of X and y does not match"

    N = X.shape[0]                      # Number of data points
    m = 10                              # Size of each mini-batch
    num_epochs = int(1e4)                 # Number of epochs
    theta = np.ones((X.shape[1], 1))    # Initialize theta with ones
    eta = lambda epoch:  0.1            # Learning rate (constant here)
    np.random.seed(4155)

    for epoch in range(1,num_epochs+1):
        batch = np.random.choice(N, m, replace=False) # Mini batch
        g = 2.0/len(y) * X.T @ ((X @ theta)-y) # Compute gradient
        theta -= eta(epoch)*g # Update theta

    return theta.ravel()





if __name__ == "__main__":
    # Get modules from project 1
    path = os.getcwd() # Current working directory
    path += '/../../Project 1/code'
    sys.path.append(path)
    from Functions import *

    #--- Create data from Franke Function ---#
    N = 5               # Number of points in each dimension
    z_noise = 0.2       # Added noise to the z-value
    n = 2               # Highest order of polynomial for X
    
    x, y, z = generate_data(N, z_noise)
    X = create_X(x, y, n)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
    mean_scale(X_train, X_test, z_train, z_test)

    #--- Regression ---#
    theta = SGD(X_train, z_train) # Stochastic Gradient Descent
    theta_OLS = OLS_regression(X_train, z_train) # OLS regression
    print("SGD: ",theta)
    print("OLS:", theta_OLS)









    # SGD()
