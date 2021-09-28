import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.utils import resample
from sklearn.model_selection import  KFold, cross_val_score

def R2(y_data, y_model):
    """
    Evluated the square error of the data.
    """
    return 1 - np.sum((y_data.ravel() - y_model.ravel())**2) / np.sum((y_data.ravel() - np.mean(y_data.ravel())) ** 2)


def MSE(y_data, y_model):
    """
    Evaluates the mean sqaured error of the data.
    """
    return np.mean((y_data.ravel() - y_model.ravel())**2)


def FrankeFunction(x, y):
    """
    Generates a set of values using Franke function.
    """
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2)**2) - 0.25 * ((9 * y - 2)**2))
    term2 = 0.75 * np.exp(-((9 * x + 1)**2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-(9 * x - 7)**2 / 4.0 - 0.25 * ((9 * y - 3)**2))
    term4 = -0.2 * np.exp(-(9 * x - 4)**2 - (9 * y - 7)**2)
    return term1 + term2 + term3 + term4


def OLS_regression(X, y):
    """
    Ordinary Least Squares
    Matrix inversion to find beta
    X (train/test): Design matrix
    y: (train/test) data output
    """
    beta = (np.linalg.pinv(X.T @ X) @ X.T @ y).ravel()

    return beta


def RIDGE_regression(X, y, lamda):
    """
    Rigde
    Matrix inversion to find beta
    X (train/test): Design matrix
    y: (train/test) data output
    lamda: parameter to avoid singular matrix
    """
    dim = X.shape[1]
    beta = np.linalg.pinv(X.T @ X + lamda * np.eye(dim)) @ X.T @ y
    return beta


def generate_2D_mesh_grid(N):
    """
    Generates 2D mesh grid (x,y)
    N: Number of uniform points
    """

    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    x, y = np.meshgrid(x, y)
    return x, y


def create_X(x, y, n):
    """
    Creates design matrix X
    for polynomials in 2D up to order n as:
    [x, y, x^2, xy, y^2, ... x^n, .. y^n]
    """
    if len(x.shape) > 1:
        x = x.reshape(x.shape[0] * x.shape[1])  # flattens x
        y = y.reshape(y.shape[0] * y.shape[1])  # flattens y

    N = len(x)
    l = int((n + 1) * (n + 2) / 2)		# Number of elements in beta
    X = np.ones((N, l))

    for i in range(1, n + 1):
        q = int((i) * (i + 1) / 2)
        for k in range(i + 1):
            X[:, q + k] = (x**(i - k)) * (y**k)
    return X


def standard_scale(z):
    """
    Returns the standard scaling of z using
    the mean of z and the standard deviation.
    """
    z_scaled = (z - np.mean(z)) / np.std(z)
    return z_scaled


def scale_design_matrix(X_train, X_test):
    """
    Scales the desing matrix with sklearns standard StandardScaler
    with the X_train set.
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Force first column of X back to 1
    X_train[:, 0] = 1.
    X_test[:, 0] = 1.

    return X_train, X_test


def generate_data(N, z_noise, seed=4155):
    """
    Generates x,y mesh grid and
    corresponding z-values from FrankeFunction
    """
    if seed != None:
        np.random.seed(seed)  # Random seed

    x, y = generate_2D_mesh_grid(N)
    z = FrankeFunction(x, y) + z_noise * np.random.randn(N, N)
    z = z.reshape(N**2, 1)  # flatten
    return x, y, z

def cross_validation(X, z, k_fold_number, method, lamda=0, include_train=False):
    kfold = KFold(n_splits=k_fold_number)
    j = 0
    z_pred_arr = np.zeros((int(np.shape(X)[0] / k_fold_number), k_fold_number))

    MSE_arr = np.zeros(k_fold_number)
    if include_train:
        MSE_arr_tilde = np.zeros(k_fold_number)

    for train_indx, test_indx in kfold.split(X):
        X_train = X[train_indx]
        z_train = z[train_indx]

        X_test = X[test_indx]
        z_test = z[test_indx]

        X_train, X_test = scale_design_matrix(X_train, X_test)
        if method == "OLS":
            beta = OLS_regression(X_train, z_train)
            z_pred = (X_test @ beta).ravel()
            if include_train:
                z_tilde = (X_train @ beta).ravel()

        elif method == "Ridge":
            beta = RIDGE_regression(X_train, z_train, lamda)
            z_pred = (X_test @ beta).ravel()
            if include_train:
                z_tilde = (X_train @ beta).ravel()

        elif method == "Lasso":
            RegLasso = linear_model.Lasso(lamda)
            RegLasso.fit(X_train, z_train)
            z_pred = RegLasso.predict(X_test)
            if include_train:
                z_tilde = RegLasso.predict(X_train)

        MSE_arr[j] = MSE(z_test.ravel(), z_pred)
        if include_train:
            MSE_arr_tilde[j] = MSE(z_train.ravel(), z_tilde)

        j += 1
    if include_train:
        return np.mean(MSE_arr), np.mean(MSE_arr_tilde)
    return np.mean(MSE_arr)


def bootstrap(X_train, X_test, z_train, z_test, B, method, lamda=0, include_train=False):
    """
    info
    """

    z_pred = np.zeros((len(z_test), B))
    if include_train:
        z_tilde = np.zeros((len(z_train), B))
    if method == "OLS":
        for i in range(B):
            X_res, z_res = resample(X_train, z_train)
            beta = OLS_regression(X_res, z_res)
            z_pred[:, i] = (X_test @ beta).ravel()
            if include_train:
                z_tilde[:, i] = (X_train @ beta).ravel()
    elif method == "Ridge":
        for i in range(B):
            X_res, z_res = resample(X_train, z_train)
            beta = RIDGE_regression(X_res, z_res, lamda)
            z_pred[:, i] = (X_test @ beta).ravel()
            if include_train:
                z_tilde[:, i] = (X_train @ beta).ravel()
    elif method == "Lasso":
        for i in range(B):
            X_res, z_res = resample(X_train, z_train)
            RegLasso = linear_model.Lasso(lamda, tol = 1e-2)
            RegLasso.fit(X_res, z_res)
            z_pred[:, i] = RegLasso.predict(X_test)
            if include_train:
                z_tilde[:, i] = RegLasso.predict(X_train)
    if include_train:
        return z_pred, z_tilde
    else:
        return z_pred
