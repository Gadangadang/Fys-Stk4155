import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from prediction_plots import plot_3D
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from Functions import *
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def compare_OLS_R_L(data, n_values, lamda_values, k_fold_number):
    MSE_OLS = np.zeros(len(n_values))
    MSE_Ridge = np.zeros((len(n_values), len(lamda_values)))
    MSE_Lasso = np.zeros((len(n_values), len(lamda_values)))
    x,y,z = data
    i = 0
    j = 0
    OLS = LinearRegression(normalize = True)
    txt_info =  "Regression analysis:"
    for n in n_values:
        X = create_X(x, y, n)
        MSE_OLS[i] = np.mean(-cross_val_score(OLS, X, z, scoring='neg_mean_squared_error', cv=k_fold_number))
        for lmb in lamda_values:
            print(f"\r{txt_info}", f"process: {100*(i*len(lamda_values) + j)/(len(lamda_values)*len(n_values)):.2f} %",end="")
            ridge = Ridge(alpha = lmb, max_iter = 1000000,normalize=True)
            lasso = Lasso(alpha = lmb,  max_iter = 1000000, normalize=True)
            MSE_Lasso[i,j] = np.mean(-cross_val_score(lasso, X, z, scoring='neg_mean_squared_error', cv=k_fold_number))
            MSE_Ridge[i,j] = np.mean(-cross_val_score(ridge, X, z, scoring='neg_mean_squared_error', cv=k_fold_number))
            j += 1
        i += 1
        j = 0
    indx1 = np.where(MSE_OLS == np.min(MSE_OLS))
    indx2 = np.where(MSE_Ridge == np.min(MSE_Ridge))
    indx3 = np.where(MSE_Lasso == np.min(MSE_Lasso))
    print(indx1)
    print(indx2)
    print(indx3)
    exit()
    print(lamda_values[indx2[1]],lamda_values[indx3[1]])
    print(n_values[indx1[0]], n_values[indx2[0]],n_values[indx3[0]])

    cmap = plt.get_cmap('RdBu')
    fig, axs = plt.subplots(1,3,figsize=(8,5))
    axs[0].plot(n_values, MSE_OLS)
    im2 = axs[1].pcolormesh(np.log(np.asarray(lamda_values)), n_values, MSE_Lasso, cmap='RdBu',shading='auto' )
    im3 = axs[2].pcolormesh(np.log(np.asarray(lamda_values)), n_values,  MSE_Ridge, cmap='RdBu',shading='auto' )
    #axs[0].scatter(indx1, MSE_OLS[indx1])
    #axs[1].scatter(indx2[0], indx2[1])
    #axs[2].scatter(indx3[0], indx3[1])
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
    z = imread("../article/Saudi.tif")[::100,::100]
    x_len, y_len = np.shape(z)
    x = np.linspace(0, x_len-1, x_len)
    y = np.linspace(0, y_len-1, y_len)
    #scaler = StandardScaler()
    #x,y = np.meshgrid(x,y)
    #z = scaler.fit_transform(z)
    z = (z)/np.max(z)

    #plot_3D("Saudi", x, y, z, "Høyde", "save_name", show = True, save = False)
    data = [x,y,z]
    lamda_values = lamda_values = np.logspace(-5, 0, 25)
    n_values = range(1,8)
    k_fold_number = 5
    compare_OLS_R_L(data, n_values, lamda_values, k_fold_number)
