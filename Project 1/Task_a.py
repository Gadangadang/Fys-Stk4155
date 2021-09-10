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

def evaluate_regression(beta, X_train, X_test, z_train, z_test):
    """
    """
    # Prediction
    ztilde = X_train @ beta
    zpredict = X_test @ beta

    MSE_train = MSE(z_train, ztilde)
    MSE_test =  MSE(z_test,zpredict)
    R2_train =  R2(z_train,ztilde)
    R2_test =   R2(z_test,zpredict)

    # alpha-% confidential interval (standard normal distribution)
    alpha = 0.95
    from scipy.stats import norm
    Z = norm.ppf(alpha + (1-alpha)/2)
    diff = z_test-zpredict
    sample_mean = np.mean(diff)
    SE = np.std(diff)/np.sqrt(len(diff))
    confidence_intervals = [sample_mean - Z*SE, sample_mean + Z*SE]

    #--- print result ---#
    print_results = True
    if print_results:
        print("#----- Error -----#")
        print("      train  |  test")
        print(f"MSE: {MSE_train:2.5f} | {MSE_test:2.5f}")
        print(f"R2 : {R2_train:2.5f} | {R2_test:2.5f}")

        print(f"\n#----- {alpha}% confidence intervals -----#")
        print(f"[{confidence_intervals[0]:.2e}, {confidence_intervals[1]:.2e}]")









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
    evaluate_regression(beta_OLS, X_train, X_test, z_train, z_test)
