import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from Functions import *

def generate_data(N, z_noise):
    """
    Generates x,y mesh grid and
    corresponding z-values from FrankeFunction
    """
    N = 100
    x,y = generate_2D_mesh_grid(N)
    z = FrankeFunction(x, y) + z_noise*np.random.randn(N,N)
    z = z.reshape(N**2) # flatten
    return x, y, z




if __name__ == "__main__":

    #--- settings ---#
    N = 100             # Number of points in each dimension
    z_noise = 0.1       # Added noise to the z-value
    n = 5               # Highest order of polynomial for X

    # Create data and set up design matrix
    x, y, z = generate_data(N, z_noise)
    X = create_X(x, y, n)

    # Split data into train and test data
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

    # Scale data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Force first column of X back to 1
    X_train_scaled[:,0] = 1.
    X_test_scaled[:,0] = 1.


    #OLS regression
    beta_OLS = OLS_regression(X_train, X_test, z_train, z_test)

    # Prediction
    ztilde = X_train @ beta
    zpredict = X_test @ beta
    
