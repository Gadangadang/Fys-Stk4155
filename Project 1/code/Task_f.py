import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from prediction_plots import plot_3D
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from Functions import *
from sklearn.preprocessing import StandardScaler

def compare_OLS_R_L(data, n_values, lamda_values, k_fold_number):
    MSE_OLS = np.zeros(len(n_values))
    MSE_Ridge = np.zeros((len(n_values), len(lamda_values)))
    MSE_Lasso = np.zeros((len(n_values), len(lamda_values)))
    x,y,z = data
    z = np.asarray(z)
    z = z - np.mean(z)

    i = 0
    j = 0
    OLS = LinearRegression()
    txt_info =  "Regression analysis:"
    for n in n_values:
        X = create_X(x, y, n)
        mean_scale(X)
        MSE_OLS[i] = np.mean(-cross_val_score(OLS, X, z, scoring='neg_mean_squared_error', cv=k_fold_number))
        for lmb in lamda_values:
            print(f"\r{txt_info}", f"process: {100*(i*len(lamda_values) + j)/(len(lamda_values)*len(n_values)):.2f} %",end="")
            ridge = Ridge(alpha = lmb, max_iter = 100000)
            lasso = Lasso(alpha = lmb,  max_iter = 100000)
            MSE_Lasso[i,j] = np.mean(-cross_val_score(lasso, X, z, scoring='neg_mean_squared_error', cv=k_fold_number))
            MSE_Ridge[i,j] = np.mean(-cross_val_score(ridge, X, z, scoring='neg_mean_squared_error', cv=k_fold_number))
            j += 1
        i += 1
        j = 0
    print(MSE_Lasso)
    cmap = plt.get_cmap('RdBu')
    fig, axs = plt.subplots(1,3,figsize=(8,5))
    axs[0].plot(n_values, MSE_OLS)
    axs[0].set(ylim=(min(MSE_OLS), 1))
    im2 = axs[1].pcolormesh(np.log(np.asarray(lamda_values)), n_values, MSE_Lasso, cmap='RdBu')
    im3 = axs[2].pcolormesh(np.log(np.asarray(lamda_values)), n_values, MSE_Ridge, cmap='RdBu')
    axs[0].set_title("OLS")
    axs[1].set_title("Lasso")
    axs[2].set_title("Ridge")
    plt.colorbar(im2, ax=axs[1])
    plt.colorbar(im3, ax=axs[2])
    fig.tight_layout()
    plt.show()



if __name__ == "__main__":
    N = 20
    z_noise = 0.2
    n = 10
    B = 100
    terrain1 = imread("../article/Norway.tif")
    z = terrain1 #[::100,::100]
    length = np.shape(z)[0]
    print(length)
    x = np.linspace(0, length-1, length)
    y = np.linspace(0, length-1, length)
    x,y = np.meshgrid(x,y)
    z = (z)/np.mean(z)

    plot_3D("Saudi", x, y, z, "Høyde", "save_name", show = True, save = False)
    exit()
    data = [x,y,z]
    lamda_values = lamda_values = np.logspace(-3, 0, 5)
    n_values = range(0,5)
    k_fold_number = 5
    compare_OLS_R_L(data, n_values, lamda_values, k_fold_number)
