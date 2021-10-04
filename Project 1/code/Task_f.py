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
from plot_set import*
from matplotlib.ticker import MaxNLocator

from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
@ignore_warnings(category=ConvergenceWarning)

def compare_OLS_R_L(data, n_values, lamda_values, k_fold_number):
    MSE_OLS = np.zeros(len(n_values))
    MSE_Ridge = np.zeros((len(n_values), len(lamda_values)))
    MSE_Lasso = np.zeros((len(n_values), len(lamda_values)))
    x,y,z = data

    OLS = LinearRegression()

    txt_info =  "Regression analysis:"
    for i in range(len(n_values)):
        X = create_X(x, y, n_values[i])
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)

        MSE_OLS[i] = np.mean(-cross_val_score(OLS, X, z, scoring='neg_mean_squared_error', cv=k_fold_number))
        for j in range(len(lamda_values)):
            print(f"\r{txt_info}: process: n = {i}/{len(n_values)-1}, lmb = {j}/{len(lamda_values)-1}", end="")

            max_iter = int(1e4)
            ridge = Ridge(alpha = lamda_values[j], max_iter = max_iter, normalize=True)
            lasso = Lasso(alpha = lamda_values[j],  max_iter = max_iter, normalize=True)
            MSE_Ridge[i,j] = np.mean(-cross_val_score(ridge, X, z, scoring='neg_mean_squared_error', cv=k_fold_number))
            MSE_Lasso[i,j] = np.mean(-cross_val_score(lasso, X, z, scoring='neg_mean_squared_error', cv=k_fold_number))
    print(" (done)")

    idx1 = np.argmin(MSE_OLS)
    idx2 = np.argwhere(MSE_Ridge == np.min(MSE_Ridge)).ravel()
    idx3 = np.argwhere(MSE_Lasso== np.min(MSE_Lasso)).ravel()

    #-- Plotting --#
    # OLS
    plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
    plt.plot(n_values, MSE_OLS, "o--")
    plt.ylim(MSE_OLS.min()*0.5,MSE_OLS.min()*15 )
    plt.title("OLS")
    plt.plot(n_values[idx1], MSE_OLS[idx1], markersize = 20, marker = "x", color = "black")
    plt.xlabel(r"$n$", fontsize=14)
    plt.ylabel(r"$\lambda$", fontsize=14)


    # Ridge
    cmap = plt.get_cmap('RdBu') # Cmap
    plt.figure(num=1, dpi=80, facecolor='w', edgecolor='k')
    levels = MaxNLocator(nbins=30).tick_values(np.min(MSE_Ridge), np.max(MSE_Ridge))
    plt.contourf(n_values,lamda_values, MSE_Ridge.T, vmin=np.min(MSE_Ridge), vmax=np.max(MSE_Ridge), cmap='RdBu',levels=levels)
    plt.plot(n_values[idx2[0]], lamda_values[idx2[1]], markersize = 20, marker = "x", color = "black")
    plt.yscale("log")
    cbar1 = plt.colorbar()
    cbar1.set_label(r"MSE", fontsize=14, rotation=270, labelpad= 20)
    plt.title("Ridge")
    plt.xlabel(r"$n$", fontsize=14)
    plt.ylabel(r"$\lambda$", fontsize=14)

    # Lasso
    plt.figure(num=2, dpi=80, facecolor='w', edgecolor='k')
    levels = MaxNLocator(nbins=30).tick_values(np.min(MSE_Lasso), np.max(MSE_Lasso))
    plt.contourf(n_values,lamda_values, MSE_Lasso.T, vmin=np.min(MSE_Lasso), vmax=np.max(MSE_Lasso), cmap='RdBu',levels=levels)
    plt.plot(n_values[idx3[0]], lamda_values[idx3[1]], markersize = 20, marker = "x", color = "black")
    plt.yscale("log")
    cbar2 = plt.colorbar()
    cbar2.set_label(r"MSE", fontsize=14, rotation=270, labelpad= 20)
    plt.title("Lasso")
    plt.xlabel(r"$n$", fontsize=14)
    plt.ylabel(r"$\lambda$", fontsize=14)
    plt.show()



def data_split(x, y, z, split = 0.2):
    split_idx = int((1-0.2)*len(z))
    shuffle_idx = np.arange(z.shape[0]*z.shape[1])
    np.random.shuffle(shuffle_idx)
    x, y, z = x[shuffle_idx], y[shuffle_idx], z[shuffle_idx]
    x_train, y_train, z_train = x[:split_idx], y[:split_idx], z[:split_idx]
    x_test, y_test, z_test = x[split_idx:], y[split_idx:], z[split_idx:]
    return x_train, y_train, z_train, x_test, y_test, z_test


if __name__ == "__main__":
    z = imread("../article/Saudi.tif")[::50,::50]
    x_len, y_len = np.shape(z)
    x = np.linspace(0, x_len-1, x_len)
    y = np.linspace(0, y_len-1, y_len)
    z = (z - np.mean(z))/np.std(z) # standard scale

    x,y = np.meshgrid(x,y)
    x = x.reshape(x.shape[0] * x.shape[1])  # flattens x
    y = y.reshape(y.shape[0] * y.shape[1])  # flattens y
    z = z.reshape(z.shape[0]**2, 1)

    # Split x,y,z in train and test
    x_train, y_train, z_train, x_test, y_test, z_test = data_split(x, y, z, split = 0.2)


    #plot_3D("Saudi", x, y, z, "Høyde", "save_name", show = True, save = False)
    data = [x_train, y_train, z_train]
    # print(np.shape(x_train))
    # exit()
    # data = [x, y, z]

    lamda_values = np.logspace(-7, -3, 7)
    n_values = range(1,9)
    k_fold_number = 5
    compare_OLS_R_L(data, n_values, lamda_values, k_fold_number)






#
