import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from Functions import *


def cross_validation(N, z_noise, k_fold_number, n):
    """
    
    """
    x, y, z = generate_data(N, z_noise, seed=2018)
    X = create_X(x, y, n)
    kfold = KFold(n_splits = k_fold_number)
    j = 0
    MSE_array = np.zeros(k_fold)

    for train_indx, test_indx in kfold.split(X):
        X_train = X[train_inds]
        z_train = z[train_inds]

        X_test = x[test_inds]
        z_test = z[test_inds]

        beta_OLS = OLS_regression(X_train, z_train)
        z_pred = (X_test @ beta_OLS).ravel()
        MSE_array[j] = MSE(z_test, z_pred)



if __name__ == "__main__":

    cross_validation(N, z_noise, k_fold_number)
