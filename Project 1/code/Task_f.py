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
    X_F = create_X(x, y, n_values[-1])
    X_train, X_test, z_train, z_test = train_test_split(
        X_F, z, test_size=0.2)
    kfold = KFold(n_splits=k_fold_number)
    txt_info =  "Regression analysis:"
    scaler = StandardScaler()
    for i in range(len(n_values)):
        l = int((n_values[i] + 1) * (n_values[i] + 2) / 2)
        X = X_train[:,:l]

        scaler.fit(X)
        X = scaler.transform(X)

        MSE_OLS[i] = np.mean(-cross_val_score(OLS, X, z_train, scoring='neg_mean_squared_error', cv=kfold))
        for j in range(len(lamda_values)):
            print(f"\r{txt_info}: process: n = {i}/{len(n_values)-1}, lmb = {j}/{len(lamda_values)-1}", end="")

            max_iter = int(1e4)
            ridge = Ridge(alpha = lamda_values[j], max_iter = max_iter, normalize=True)
            lasso = Lasso(alpha = lamda_values[j],  max_iter = max_iter, normalize=True)
            MSE_Ridge[i,j] = np.mean(-cross_val_score(ridge, X, z_train, scoring='neg_mean_squared_error', cv=kfold))
            MSE_Lasso[i,j] = np.mean(-cross_val_score(lasso, X, z_train, scoring='neg_mean_squared_error', cv=kfold))
    print(" (done)")


    idx1 = np.argmin(MSE_OLS)
    idx2 = np.argwhere(MSE_Ridge == np.min(MSE_Ridge)).ravel()
    idx3 = np.argwhere(MSE_Lasso== np.min(MSE_Lasso)).ravel()

    n = n_values[idx1]
    l = int((n + 1) * (n + 2) / 2)
    X = X_train[:,:l]
    scaler.fit(X)
    X = scaler.transform(X)
    scaler.fit(X_test)
    X_test = scaler.transform(X_test)
    OLS.fit(X,z_train)
    OLS_predict  = OLS.predict(X_test)

    n, lmb = n_values[idx2[0]], lamda_values[idx2[1]]
    l = int((n + 1) * (n + 2) / 2)
    X = X_train[:,:l]
    scaler.fit(X)
    X = scaler.transform(X)
    scaler.fit(X_test)
    X_test = scaler.transform(X_test)
    ridge = Ridge(alpha = lmb, max_iter = max_iter, normalize=True).fit(X,z_train)
    Ridge_predict = ridge.predict(X_test)

    n, lmb = n_values[idx3[0]], lamda_values[idx3[1]]
    l = int((n + 1) * (n + 2) / 2)
    X = X_train[:,:l]
    scaler.fit(X)
    X = scaler.transform(X)
    scaler.fit(X_test)
    X_test = scaler.transform(X_test)
    lasso = Lasso(alpha = lmb, max_iter = max_iter, normalize=True).fit(X,z_train)
    Lasso_predict = lasso.predict(X_test)

    MSE_OLS, MSE_Ridge, MSE_Lasso = MSE(z_test, OLS_predict), MSE(z_test, Ridge_predict), MSE(z_test, Lasso_predict)
    print(f"MSE for varying methods: OLS = {MSE_OLS:.5f} -- Ridge = {MSE_Ridge:.5f} -- Lasso = {MSE_Lasso:.5f} ")
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
    return X_test, z_test, OLS_predict, Ridge_predict, Lasso_predict

def plot_approx(X_test, z_test, OLS_predict, Ridge_predict, Lasso_predict):
    lenght = int(np.sqrt(len(z_test)))
    plt.scatter(X_test, z_test)
    #plt.imshow(z_test.reshape(lenght, lenght))
    #plt.colorbar()
    exit()

    fig = plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                           linewidth=0.2, antialiased=False)
    ax.set_title("Real data")
    ax = fig.add_subplot(2, 2, 2, projection='3d')
    surf = ax.plot_surface(x, y, OLS_predict, cmap=cm.coolwarm,
                           linewidth=0.2, antialiased=False)
    ax.set_title("OLS-predict")
    ax = fig.add_subplot(2, 2, 2, projection='3d')
    surf = ax.plot_surface(x, y, Ridge_predict, cmap=cm.coolwarm,
                           linewidth=0.2, antialiased=False)
    ax.set_title("Ridge-predict")
    ax = fig.add_subplot(2, 2, 2, projection='3d')
    surf = ax.plot_surface(x, y, Lasso_predict, cmap=cm.coolwarm,
                           linewidth=0.2, antialiased=False)
    ax.set_title("Lasso-predict")
    plt.show()



if __name__ == "__main__":
    z = imread("../article/Saudi.tif")[::100,::100]
    x_len, y_len = np.shape(z)
    x = np.linspace(0, x_len-1, x_len)
    y = np.linspace(0, y_len-1, y_len)
    z = (z - np.mean(z))/np.std(z) # standard scale

    x,y = np.meshgrid(x,y)
    x_flat = x.reshape(x.shape[0] * x.shape[1])  # flattens x
    y_flat = y.reshape(y.shape[0] * y.shape[1])  # flattens y
    z_flat = z.reshape(z.shape[0]**2, 1)

    #plot_3D("Saudi", x, y, z, "Høyde", "save_name", show = True, save = False)
    data = [x_flat, y_flat, z_flat]

    lamda_values = np.logspace(-8, -1, 8)
    n_values = range(1,8)
    k_fold_number = 5
    X_test, z_test, OLS_predict, Ridge_predict, Lasso_predict = compare_OLS_R_L(data, n_values, lamda_values, k_fold_number)

    #plot_approx(X_test, z_test, OLS_predict, Ridge_predict, Lasso_predict)





#
