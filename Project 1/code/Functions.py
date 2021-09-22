import numpy as np


def R2(y_data, y_model):
    """
    Evluated the square error of the data.
    """
    return 1 - np.sum((y_data - y_model)**2) / np.sum((y_data - np.mean(y_data)) ** 2)

def MSE(y_data,y_model):
    """
    Evaluates the mean sqaured error of the data.
    """
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

def FrankeFunction(x,y):
    """
    Generates a set of values using Franke function.
    """
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def OLS_regression(X, y):
    """
    Ordinary Least Squares
    Matrix inversion to find beta
    X (train/test): Design matrix
    y: (train/test) data output
    """
    beta = np.linalg.pinv(X.T @ X) @ X.T @ y
    return beta


def generate_2D_mesh_grid(N):
    """
    Generates 2D mesh grid (x,y)
    N: Number of uniform points
    """


    x = np.linspace(0,1, N)
    y = np.linspace(0,1, N)
    x, y = np.meshgrid(x,y)
    return x, y

def create_X(x, y, n ):
    """
    Creates design matrix X
    for polynomials in 2D up to order n as:
    [x, y, x^2, xy, y^2, ... x^n, .. y^n]
    """
    if len(x.shape) > 1:
        x = x.reshape(x.shape[0]*x.shape[1]) # flattens x
        y = y.reshape(y.shape[0]*y.shape[1]) # flattens y

    N = len(x)
    l = int((n+1)*(n+2)/2)		# Number of elements in beta
    X = np.ones((N,l))

    for i in range(1,n+1):
    	q = int((i)*(i+1)/2)
    	for k in range(i+1):
    		X[:,q+k] = (x**(i-k))*(y**k)
    return X

def standard_scale(z):
    """
    Returns the standard scaling of z using
    the mean of z and the standard deviation.
    """
    z_scaled = (z-np.mean(z))/np.std(z)
    return z_scaled

def scale_design_matrix(X_train, X_test):
    """
    Scales the desing matrix
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Force first column of X back to 1
    X_train[:,0] = 1.
    X_test[:,0] = 1.

    return X_train, X_test

def generate_data(N, z_noise, seed = 4155):
    """
    Generates x,y mesh grid and
    corresponding z-values from FrankeFunction
    """
    if seed != None:
        np.random.seed(seed) # Random seed


    x,y = generate_2D_mesh_grid(N)
    z = FrankeFunction(x, y) + z_noise*np.random.randn(N,N)
    z = z.reshape(N**2) # flatten
    return x, y, z
