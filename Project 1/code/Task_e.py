import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from Functions import *
from matplotlib.ticker import MaxNLocator
from sklearn.utils import resample
from plot_set import *  # Specifies plotting settings
from Task_d import bias_variance_tradeoff_lamda





if __name__ == "__main__":
    #--- settings ---#
    N =  30          # Number of points in each dimension
    z_noise = 0.2     # Added noise to the z-value
    n = 14                # Highest order of polynomial for X
    B = 100             # Number of training points
    method = "Lasso"
    lamba = np.logspace(-3,2,4)

    bias_variance_tradeoff_lamda(lamba, N, z_noise, n, B, method, plot=True)
    #bias_variance_tradeoff(N, z_noise, n, B, method, plot=True)
