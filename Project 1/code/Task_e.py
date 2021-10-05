import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from Functions import *
from matplotlib.ticker import MaxNLocator
from sklearn.utils import resample
from plot_set import *  # Specifies plotting settings
from Task_b2 import bias_variance_tradeoff
from Task_c import compaire_CV_B
from Task_d import MSE_bootstrap

if __name__ == "__main__":
    #--- settings ---#
    N = 15          # Number of points in each dimension
    z_noise = 0.2     # Added noise to the z-value
    n = 24                # Highest order of polynomial for X
    B = "N"            # Number of training points
    method = "Lasso"
    lamda = np.logspace(-3, -1, 4)
    k_fold_number = 5
    B = 100

    #bias_variance_tradeoff(N, z_noise, n, "N", method, lamda, plot=True)

    compaire_CV_B(N, z_noise, n, N * N, k_fold_number, method, lamda=1e-3)
    
    N = 30
    z_noise = 0.2
    n = 14
    B = 100

    method = "Lasso"

    lmb = 1e-4
    compaire_CV_B(generate_data(N, z_noise, seed=4155), n, B, k_fold_number, method, lamda = lmb)

    lmb = 1e-3
    compaire_CV_B(generate_data(N, z_noise, seed=4155), n, B, k_fold_number, method, lamda = lmb)

    lmb = 1e-2
    compaire_CV_B(generate_data(N, z_noise, seed=4155), n, B, k_fold_number, method, lamda = lmb)

    lmb = 1e2
    compaire_CV_B(generate_data(N, z_noise, seed=4155), n, B, k_fold_number, method, lamda = lmb)
