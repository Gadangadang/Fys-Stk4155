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
        """Func explanation"""
        #Setup model
        model = tf.model.Sequential([
            tf.keras.layers.Dense(10, activation="sigmoid", input_shape=(5,)),
            tf.keras.layers.Dense(10, activation="sigmoid"),
            tf.keras.layers.Dense(3, activation="sigmoid"),
        ])

        #
        #model.compile(loss = "cost_function", optimizer="adam", metrics=["MSE"])
        ...

    # The right side of the ODE:
    def f(self, point):
        return 0.

    def gradient(self, model, inputs, targets):
      with tf.GradientTape() as tape:
        loss_value = cost_function(model, inputs, targets, training=True)
      return loss_value, tape.gradient(loss_value, model.trainable_variables)

    def cost_function(self, model, inputs, target, training=True):
        self.model_tf = model(x, training=training)
        y_ = self.g_trial()
        y = target

        return (y_ - y)**2


    def g_trial(self):
        """
        g_trial(x, t) = h_1(x, t) + h_2(x,t)N(x,t,P)
        h_1 and h_2 are functions to control boundary and inital conditions
        """
        return (1-self.time)*self.u(self.x) + self.x*(1-self.x)*self.time*self.model_tf

def g_analytic(x, t):
    #Analytic solution to function
    return np.exp(-np.pi**2*t)*np.sin(np.pi*x)

def u(x):
    #Initial condition
    return np.sin(np.pi*x)


if __name__ == "__main__":
    #Check tensorflow version and eager execution
    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))
