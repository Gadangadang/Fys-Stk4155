import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



def linalg_approx():
    beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
    ytilde = X_train @ beta
    ypredict = X_test @ beta
