import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression





if __name__ == "__main__":
    from Functions import *
    x,y = generate_2D_mesh_grid()


    exit()
    noise = 0.1*np.random.randn(N,N)

    z = FrankeFunction(x, y) + 0.1*np.random.randn(N,N)
    z = z.reshape(N**2) #Flatten
    X = create_X(x, y, n=n)
