import numpy as np


def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model)**2) / np.sum((y_data - np.mean(y_data)) ** 2)

#Mean sqaured error
def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n


def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

#Matrix inversion method.
def matrix_inv(x_values, y_values):
    beta = np.linalg.inv(x_values.T @ x_values) @ x_values.T @ y_values
    ytilde = x_values @ beta
    return ytilde


def generate_data():
    """
    Generates x and y values with normal distributed noise.
    """
    np.random.seed(4155)
    x = np.random.rand(100,1)
    y = 2.0+5*x*x+0.1*np.random.randn(100,1)
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
