import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from prediction_plots import plot_3D
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from Functions import *


def compare_OLS_R_L(data, n_values, lamda_values, k_fold_number):
    MSE_OLS = np.zeros(len(n_values)+1)
    MSE_Ridge = np.zeros((len(n_values)+1, len(lamda_values)+1))
    MSE_Lasso = np.zeros((len(n_values)+1, len(lamda_values)+1))
    x,y,z = data

    i = 0
    j = 0
    OLS = LinearRegression()

    for n in n_values:
        X = create_X(x, y, n)
        X_train, X_test, z_train, z_test = train_test_split(
            X, z, test_size=0.2)
        X_train, X_val, z_train, z_val = train_test_split(
            X, z, test_size=0.2)
        poly = PolynomialFeatures(degree = 6)
        train_set = poly.fit_transform(X_train)
        MSE_OLS[i] = np.mean(-cross_validate(OLS, train_set, z_val,scoring='neg_mean_squared_error', cv=k_fold_number))
        for lmb in lamda_values:
            ridge = Ridge(alpha = lmb)
            lasso = Lasso(alpha = lmb)
            MSE_Lasso[i,j] = np.mean(-cross_validate(lasso, train_set, z_val,  scoring='neg_mean_squared_error', cv=k_fold_number))
            MSE_Ridge[i,j] = np.mean(-cross_validate(ridge, train_set, z_val,  scoring='neg_mean_squared_error', cv=k_fold_number))
            j += 1
        i += 1
    plt.imshow(MSE_Lasso)
    plt.show()



if __name__ == "__main__":
    N = 30
    z_noise = 0.2
    n = 14
    B = 100
    terrain1 = imread("../article/Dagestan.tif")
    length = np.shape(terrain1)[0]
    x = np.linspace(0, length-1, length)
    y = np.linspace(0, length-1, length)
    method = "OLS"
    data = [x,y,terrain1]
    lamda_values = lamda_values = np.logspace(-3, 2, 20)
    n_values = range(0,21)
    k_fold_number = 5
    compare_OLS_R_L(data, n_values, lamda_values, k_fold_number)
