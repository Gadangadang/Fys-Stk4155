import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from prediction_plots import plot_3D, plot_3D_shuffled
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from Functions import *
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from plot_set import*
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl

# from sklearn.utils.testing import ignore_warnings
# from sklearn.exceptions import ConvergenceWarning


def find_best_test_size(wanted_test_size, z):
    """
    Find best test size which allow a even grid
    when reshaping back


    Args:
        wanted_test_size ([Float]): Split size for train/test
        z                  (Array): Array with data, to be split

    Returns:
        Float: Split number
        Int: Length to split z
    """
    closets_N = np.int(np.round(np.sqrt((len(z) * wanted_test_size))))
    z_test_len = closets_N**2
    split = z_test_len / len(z)
    return split, z_test_len


# @ignore_warnings(category=ConvergenceWarning)
def compare_OLS_R_L(data, n_values, lamda_values, k_fold_number, max_iter, isFranke=False):
    """Gridsearch function to find optimal paramters lambda and n

    Args:
        data              (Array): Array containing dataset x, y, z
        n_values           (List): List containing range of complexity numbers
        lamda_values      (Array): Logscale lambda values
        k_fold_number       (Int): Number of k folds
        max_iter            (Int): Max number of iterations for CV if no convergence
        isFranke (bool, optional): Choice if Franke dataset or not. Defaults to False.

    Returns:
        Array: Set of ideal n complexity values
        Array: Set of ideal lamda values
    """
    MSE_OLS = np.zeros(len(n_values))
    MSE_Ridge = np.zeros((len(n_values), len(lamda_values)))
    MSE_Lasso = np.zeros((len(n_values), len(lamda_values)))
    x, y, z = data

    # X and scaling
    X_F = create_X(x, y, n_values[-1])
    if isFranke:
        mean_scale(z, X_F)
    else:
        scaler = StandardScaler()
        scaler.fit(X_F)
        X_F = scaler.transform(X_F)
        z = (z - np.mean(z)) / np.std(z)  # standard scale
    OLS = LinearRegression()
    kfold = KFold(n_splits=k_fold_number)
    txt_info = "Regression analysis:"
    for i in range(len(n_values)):
        l = int((n_values[i] + 1) * (n_values[i] + 2) / 2)
        X = X_F[:, :l]

        MSE_OLS[i] = np.mean(-cross_val_score(OLS, X, z,
                                              scoring='neg_mean_squared_error', cv=kfold))
        for j in range(len(lamda_values)):
            print(
                f"\r{txt_info}: process: n = {i}/{len(n_values)-1}, lmb = {j}/{len(lamda_values)-1}", end="")

            ridge = Ridge(
                alpha=lamda_values[j], max_iter=max_iter, normalize=True).fit(X, z)
            lasso = Lasso(
                alpha=lamda_values[j],  max_iter=max_iter, normalize=True).fit(X, z)
            MSE_Ridge[i, j] = np.mean(-cross_val_score(ridge,
                                                       X, z, scoring='neg_mean_squared_error', cv=kfold))
            MSE_Lasso[i, j] = np.mean(-cross_val_score(lasso,
                                                       X, z, scoring='neg_mean_squared_error', cv=kfold))
    print(" (done)")

    idx1 = np.argmin(MSE_OLS)
    idx2 = np.argwhere(MSE_Ridge == np.min(MSE_Ridge)).ravel()
    idx3 = np.argwhere(MSE_Lasso == np.min(MSE_Lasso)).ravel()

    #-- Plotting --#
    # OLS
    fig_OLS = plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
    plt.plot(n_values, MSE_OLS, "o--")
    #plt.ylim(MSE_OLS.min()*0.5,MSE_OLS.min()*15 )
    plt.title("OLS")
    plt.scatter(n_values[idx1], MSE_OLS[idx1],  s=100,
                linewidths=2, marker="x", color="black", label="min MSE")
    plt.xlabel(r"$n$", fontsize=14)
    plt.ylabel(r"MSE", fontsize=14)
    plt.legend(fontsize=13)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig_OLS.savefig(
        "../article/figures/real_data_best_OLS_map.pdf", bbox_inches="tight")
    plt.show()

    # Colorbar settings
    cmap = mpl.cm.RdBu

    # Ridge
    fig_Ridge = plt.figure(num=1, dpi=80, facecolor='w', edgecolor='k')
    levels = MaxNLocator(nbins=30).tick_values(
        np.min(MSE_Ridge), np.max(MSE_Ridge))
    plt.contourf(n_values, lamda_values, MSE_Ridge.T, vmin=np.min(
        MSE_Ridge), vmax=np.max(MSE_Ridge), cmap='RdBu', levels=levels)
    plt.scatter(n_values[idx2[0]], lamda_values[idx2[1]], s=100,
                linewidths=2, marker="x", color="black", label="min MSE")
    plt.yscale("log")
    norm = mpl.colors.Normalize(vmin=np.min(MSE_Ridge), vmax=np.max(MSE_Ridge))
    cbar1 = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))
    cbar1.set_label(r"MSE", fontsize=14, rotation=270, labelpad=20)
    plt.title("Ridge")
    plt.xlabel(r"$n$", fontsize=14)
    plt.ylabel(r"$\lambda$", fontsize=14)
    plt.legend(fontsize=13)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig_Ridge.savefig(
        "../article/figures/real_data_best_Ridge_map.pdf", bbox_inches="tight")

    # Lasso
    fig_Lasso = plt.figure(num=2, dpi=80, facecolor='w', edgecolor='k')
    levels = MaxNLocator(nbins=30).tick_values(
        np.min(MSE_Lasso), np.max(MSE_Lasso))
    plt.contourf(n_values, lamda_values, MSE_Lasso.T, vmin=np.min(
        MSE_Lasso), vmax=np.max(MSE_Lasso), cmap='RdBu', levels=levels)
    plt.scatter(n_values[idx3[0]], lamda_values[idx3[1]], s=100,
                linewidths=2, marker="x", color="black", label="min MSE")
    plt.yscale("log")
    norm = mpl.colors.Normalize(vmin=np.min(MSE_Lasso), vmax=np.max(MSE_Lasso))
    cbar2 = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))
    cbar2.set_label(r"MSE", fontsize=14, rotation=270, labelpad=20)
    plt.title("Lasso")
    plt.xlabel(r"$n$", fontsize=14)
    plt.ylabel(r"$\lambda$", fontsize=14)
    plt.legend(fontsize=13)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig_Lasso.savefig(
        "../article/figures/real_data_best_Lasso_map.pdf", bbox_inches="tight")

    plt.show()

    # Save best hyper-parameter in order OLS, Ridge, Lasso
    best_n = [n_values[idx1], n_values[idx2[0]], n_values[idx3[0]]]
    best_lmd = [np.nan, lamda_values[idx2[1]], lamda_values[idx3[1]]]

    return np.array(best_n), np.array(best_lmd)


def evaluate_best_model(data_train, data_test, best_n, best_lmd, max_iter):
    """Trains the best models given the optimized hyperparameters

    Args:
        data_train (Array): Array containing all train data for x, y, z
        data_test (Array): Array containing all test data for x, y, z
        best_n (List): List containing the optimal order of complexity
        best_lmd (Array): Array containing the optimal lambda value
        max_iter (Int): Number of iteration if no convergence is found

    Returns:
        Array: Prediction for OLS
        Array: Prediction for Ridge
        Array: Prediction for Lasso
    """
    x_train, y_train, z_train = data_train
    x_test, y_test, z_test = data_test

    #--- X and scaling ---#
    scaler = StandardScaler()

    # Train
    X_F_train = create_X(x_train, y_train, best_n.max())
    n = best_n.max()

    scaler.fit(X_F_train)
    X = scaler.transform(X_F_train)
    z_train = (z_train - np.mean(z_train)) / np.std(z_train)

    # Test
    X_F_test = create_X(x_test, y_test, best_n.max())
    scaler.fit(X_F_test)
    X = scaler.transform(X_F_test)
    z_test_mean, z_test_std = np.mean(z_test), np.std(z_test)
    z_test = (z_test - z_test_mean) / z_test_std

    # OLS
    OLS = LinearRegression()
    n = best_n[0]
    l = int((n + 1) * (n + 2) / 2)
    OLS.fit(X_F_train[:, :l], z_train)
    OLS_predict = OLS.predict(X_F_test[:, :l])

    # Ridge
    n, lmb = best_n[1], best_lmd[1]
    l = int((n + 1) * (n + 2) / 2)
    ridge = Ridge(alpha=lmb, max_iter=max_iter, normalize=True).fit(
        X_F_train[:, :l], z_train)
    Ridge_predict = ridge.predict(X_F_test[:, :l])
    print(np.mean(-cross_val_score(ridge,
                                   X_F_test[:, :l], z_test, scoring='neg_mean_squared_error', cv=5)))
    print(np.mean(-cross_val_score(ridge,
                                   X_F_train[:, :l], z_train, scoring='neg_mean_squared_error', cv=5)))

    # Lasso
    n, lmb = best_n[2], best_lmd[2]
    l = int((n + 1) * (n + 2) / 2)

    lasso = Lasso(alpha=lmb, max_iter=max_iter, normalize=True).fit(
        X_F_train[:, :l], z_train)
    Lasso_predict = lasso.predict(X_F_test[:, :l])

    MSE_OLS_test, MSE_Ridge_test, MSE_Lasso_test = MSE(z_test, OLS_predict), MSE(
        z_test, Ridge_predict), MSE(z_test, Lasso_predict)
    print(
        f"MSE for varying methods: OLS = {MSE_OLS_test:.5f} -- Ridge = {MSE_Ridge_test:.5f} -- Lasso = {MSE_Lasso_test:.5f} ")

    # Rescale predictions
    OLS_predict = OLS_predict * z_test_std + z_test_mean
    Ridge_predict = Ridge_predict * z_test_std + z_test_mean
    Lasso_predict = Lasso_predict * z_test_std + z_test_mean

    return OLS_predict, Ridge_predict, Lasso_predict


def plot_predictions(data_test, OLS_predict, Ridge_predict, Lasso_predict):
    """Plots the 3D predictions from our gridsearch and optimized models

    Args:
        data_test     (Array): Containes the test data for x, y, z
        OLS_predict   (Array): Prediction model for OLS
        Ridge_predict (Array): Prediction model for Ridge
        Lasso_predict (Array): Prediction model for Lasso
    """
    x_test, y_test, z_test = data_test

    # 3D plots
    show = True
    save = True
    plot_3D_shuffled("Test data", x_test, y_test,
                     z_test.ravel(), "z", "test_data", show, save)
    plot_3D_shuffled("OLS prediction", x_test, y_test,
                     OLS_predict.ravel(), "z", "OLS_predict", show, save)
    plot_3D_shuffled("Ridge prediction", x_test, y_test,
                     Ridge_predict.ravel(), "z", "Ridge_predict", show, save)
    plot_3D_shuffled("Lasso prediction", x_test, y_test,
                     Lasso_predict.ravel(), "z", "Lasso_predict", show, save)


def train_test_split_data(x_flat, y_flat, z_flat, split):
    """Splits the given data in part of our choice

    Args:
        x_flat (Array): Flattened array with x values
        y_flat (Array): Flattened array with y values
        z_flat (Array): Flattened array with z values
        split  (Float): Percentage to split in decimal points

    Returns:
        Array: Training model for x
        Array: Training model for y
        Array: Training model for z
        Array: Test model for x
        Array: Test model for y
        Array: Test model for z

    """
    split, z_test_len_exp = find_best_test_size(split, z_flat)
    split_idx = int((1 - split) * len(z_flat))
    shuffle_idx = np.arange(len(z_flat))  # Assumes quadratic grid
    np.random.shuffle(shuffle_idx)

    x_flat, y_flat, z_flat = x_flat[shuffle_idx], y_flat[shuffle_idx], z_flat[shuffle_idx]
    x_train, y_train, z_train = x_flat[:split_idx], y_flat[:split_idx], z_flat[:split_idx]
    x_test, y_test, z_test = x_flat[split_idx:
                                    ], y_flat[split_idx:], z_flat[split_idx:]

    if len(z_test) == z_test_len_exp:
        pass
    else:
        print(f"z_test did not have wanted length of {z_test_len}\
        but had length {len(z_test)}")
    return x_train, y_train, z_train, x_test, y_test, z_test


if __name__ == "__main__":
    z = imread("../article/Saudi.tif")[::100, ::100]
    x_len, y_len = np.shape(z)
    x = np.linspace(0, x_len - 1, x_len)
    y = np.linspace(0, y_len - 1, y_len)
    isFranke = True

    x, y = np.meshgrid(x, y)
    x_flat = x.reshape(x.shape[0] * x.shape[1])  # flattens x
    y_flat = y.reshape(y.shape[0] * y.shape[1])  # flattens y
    z_flat = z.reshape(z.shape[0]**2, 1)
    if isFranke:
        z_noise, N = 0.2, 50
        x, y = generate_2D_mesh_grid(N)
        z = FrankeFunction(x, y) + z_noise * np.random.randn(N, N)
        z_flat = z.reshape(N**2, 1)
        x_flat = x.reshape(x.shape[0] * x.shape[1])  # flattens x
        y_flat = y.reshape(y.shape[0] * y.shape[1])  # flattens y
    x_train, y_train, z_train, x_test, y_test, z_test = train_test_split_data(
        x_flat, y_flat, z_flat, split=0.2)
    data_train = [x_train, y_train, z_train]
    data_test = [x_test, y_test, z_test]


    lamda_values = np.logspace(-6, -1, 5)
    n_values = range(10, 20)
    k_fold_number = 5
    max_iter = int(2e6)
    best_n, best_lmd = compare_OLS_R_L(
        data_train, n_values, lamda_values, k_fold_number, max_iter)

    OLS_predict, Ridge_predict, Lasso_predict = evaluate_best_model(
        data_train, data_test, best_n, best_lmd, max_iter)
    plot_predictions(data_test, OLS_predict, Ridge_predict, Lasso_predict)


#
