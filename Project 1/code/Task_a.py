import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from Functions import *


def confidence_interval(beta, X):
    """
    Calculates the confidence interval for a
    given beta.
    """
    alpha = 0.95
    Z = norm.ppf(alpha + (1 - alpha) / 2)  # Calculate Z

    SE_i = np.linalg.inv(X.T @ X).diagonal()  # Find the variance
    # Zip the interval.
    conf_int = np.dstack((beta - Z * SE_i, beta + Z * SE_i))[0]
    uncertainty = Z * SE_i
    uncertainty_print = f"Beta    Uncertainty \n"
    for i in range(len(beta)):
        uncertainty_print += f"{beta[i]:4.2g} +- {uncertainty[i]:2.1g}\n"
    return conf_int, uncertainty_print


def evaluate_regression(beta, X_train, X_test, z_train, z_test):
    """
    """
    # Prediction
    ztilde = X_train @ beta
    zpredict = X_test @ beta

    MSE_train = MSE(z_train, ztilde)
    MSE_test = MSE(z_test, zpredict)
    R2_train = R2(z_train, ztilde)
    R2_test = R2(z_test, zpredict)

    # alpha-% confidential interval (standard normal distribution)
    alpha = 0.95
    from scipy.stats import norm
    conf_int_train, beta_uncertainty_print_train = confidence_interval(
        beta, X_train)
    conf_int_test, beta_uncertainty_print_test = confidence_interval(
        beta, X_test)
    #--- print result ---#
    print_results = True
    if print_results:
        print("#----- Error -----#")
        print("      train  |  test")
        print(f"MSE: {MSE_train:2.5f} | {MSE_test:2.5f}")
        print(f"R2 : {R2_train:2.5f} | {R2_test:2.5f}")

        print(f"\n#----- {alpha}% confidence intervals -----#")

        print(f"\n__Training-set__")
        # print(conf_int_train)
        print(beta_uncertainty_print_train)
        print(f"\n__Test-set__")
        # print(conf_int_test)
        print(beta_uncertainty_print_test)


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

    X_train, X_test = scale_design_matrix(
        X_train, X_test)  # Scales X_train and X_test

    # OLS regression
    beta_OLS = OLS_regression(X_train, z_train)
    evaluate_regression(beta_OLS, X_train, X_test, z_train, z_test)
