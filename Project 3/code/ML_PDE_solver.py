import tensorflow as tf
import matplotlib.pyplot as plt
import autograd.numpy as np
from autograd import jacobian,hessian,grad

from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.layers import Dense, Input



class PDE_ml_solver:
    def __init__(self, L, T, dx, dt, epochs, I):
        Nx = int(L/dx)
        Nt = int(T/dt)
        self.x = np.linspace(0, L, Nx)
        self.t = np.linspace(0, T, Nt)
        self.I = I
        self.num_epochs = epochs
        self.g_t_jacobian_func = jacobian(self.g_trial,0)
        self.g_t_hessian_func = hessian(self.g_trial,0)
        self.optimizer = optimizers.SGD(learning_rate=0.01)

    def get_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation="sigmoid", input_shape=(1,2)),
            tf.keras.layers.Dense(10, activation="sigmoid"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])
        #model.compile(loss = "cost_function", optimizer="adam", metrics=["MSE"])
        return model

    def tf_run(self):
        train_loss_results = []
        train_accuracy_results = []
        model = self.get_model()
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.MeanSquaredError()
        
        for epoch in range(self.num_epochs):
            loss_value, grads = self.grad(model)
            self.optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Track progress
            epoch_loss_avg.update_state(loss_value)  # Add current batch loss
            epoch_accuracy.update_state(y, model(x, training=True))

            train_loss_results.append(epoch_loss_avg.result())
            train_accuracy_results.append(epoch_accuracy.result())
        return train_loss_results, train_accuracy_results

    def grad(self,model):

        with tf.GradientTape() as tape:
            loss_value = self.cost_function()
            return loss_value, tape.gradient(loss_value, model.trainable_variables)


    def cost_function(self,):
        """
        Calculate derivatives.
        """
        x, t  = tf.convert_to_tensor(self.x), tf.convert_to_tensor(self.t)

        with tf.GradientTape() as tape1:
            tape1.watch(x)
            with tf.GradientTape(persistent=True) as tape2:
                tape1.watch([x,t])
                g_trial = self.g_trial()
            g_t = tape1.gradient(g_trial, t)
            g_x = tape1.gradient(g_trial, x)

        g_xx = tape2.gradient(g_x, x)
        residual =  g_xx - g_t
        MSE = np.mean(residual)
        return MSE




    def g_trial(self, model):
        """
        g_trial(x, t) = h_1(x, t) + h_2(x,t)N(x,t,P)
        h_1 and h_2 are functions to control boundary and inital conditions
        """
        x,t = point
        point = np.array([x,t]).reshape(1,2)
        return (1-self.t)*self.I(x) + x\
        *(1-x)*self.t*model(point,training=True)

def g_analytic(x, t):
    #Analytic solution to function
    return np.exp(-np.pi**2*t)*I(x)

def I(x):
    #Initial condition
    return np.sin(np.pi*x)


if __name__ == "__main__":
    #Check tensorflow version and eager execution
    L  = 1
    T = 0.5
    dx = 1/100
    dt = 0.5*dx**2
    epochs = 10
    ML = PDE_ml_solver(L, T, dx, dt, epochs, I)
    loss, acc = ML.tf_run()

    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))
