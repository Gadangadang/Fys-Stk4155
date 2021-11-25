import tensorflow as tf
import matplotlib.pyplot as plt
import autograd.numpy as np
from autograd import jacobian,hessian,grad

# This allows using whichever optimiser we want (sgd,adam,RMSprop)
from tensorflow.keras import optimizers, regularizers
# This allows defining the characteristics of a particular layer
from tensorflow.keras.layers import Dense, Input
# This allows appending layers to existing models



class PDE_ml_solver:
    def __init__(self, X, y, g_analytical, x, t, initial_func):
        self.X = X
        self.target = y
        self.g_analytical = g_analytical
        self.x = x
        self.time = t
        self.u = initial_func

    def tf_run(self):
        model = tf.model.Sequential([
            tf.keras.layers.Dense(10, activation="sigmoid", input_shape=(5,)),
            tf.keras.layers.Dense(10, activation="sigmoid"),
            tf.keras.layers.Dense(3, activation="sigmoid"),
        ])

        ...

    # The right side of the ODE:
    def f(self, point):
        return 0.

    def cost_function(self, P, x, t):
        cost_sum = 0

        g_t_jacobian_func = jacobian(self.g_trial)
        g_t_hessian_func = hessian(self.g_trial)
        
        for x_i in self.x:
            for t_i in self.time:
                point = np.array([x_i,t_i])

                g_t = g_trial(point,P)
                g_t_jacobian = g_t_jacobian_func(point,P)
                g_t_hessian = g_t_hessian_func(point,P)

                g_t_dt = g_t_jacobian[1]
                g_t_d2x = g_t_hessian[0][0]

                func = self.f(point)

                err_sqr = ( (g_t_dt - g_t_d2x) - func)**2
                cost_sum += err_sqr

        return cost_sum /( np.size(x)*np.size(t) )

    def g_trial(self):
        """
        g_trial(x, t) = h_1(x, t) + h_2(x,t)N(x,t,P)
        h_1 and h_2 are functions to control boundary and inital conditions
        """
        return (1-self.time)*self.u(self.x) + self.x*(1-self.x)*self.time*self.model_tf

def g_analytic(x, t):
    return np.exp(-np.pi**2*t)*np.sin(np.pi*x)

def u(x):
    return np.sin(np.pi*x)


if __name__ == "__main__":
    #Check tensorflow version and eager execution
    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))
