import tensorflow as tf
import matplotlib.pyplot as plt
import autograd.numpy as np
from autograd import jacobian,hessian,grad

# This allows using whichever optimiser we want (sgd,adam,RMSprop)
from tensorflow.keras import optimizers, regularizers
# This allows defining the characteristics of a particular layer
from tensorflow.keras.layers import Dense, Input
# This allows appending layers to existing models
from tensorflow.keras.models import Sequential


class PDE_ml_solver:
    def __init__(self, X, y, g_analytical, x, t, initial_func):
        self.X = X
        self.target = y
        self.g_analytical = g_analytical
        self.x = x
        self.time = t
        self.u = initial_func

    def tf_run():
        ...

    def cost_function(self, P, x, t):
        ...

    def g_trial(self):
        """
        g_trial(x, t) = h_1(x, t) + h_2(x,t)N(x,t,P)
        """
        return (1-self.time)*self.u(self.x) +

def g_analytic(x, t):
    return np.exp(-np.pi**2*t)*np.sin(np.pi*x)

def u(x):
    return np.sin(np.pi*x)


if __name__ == "__main__":
    ...
