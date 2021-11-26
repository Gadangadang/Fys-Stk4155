import tensorflow as tf
import matplotlib.pyplot as plt
import autograd.numpy as np
from autograd import jacobian, hessian, grad

from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.layers import Dense, Input


class PDE_ml_solver:
    def __init__(self, L, T, dx, dt, epochs, I):
        Nx = int(L / dx)
        Nt = int(T / dt)
        self.x = tf.convert_to_tensor(np.linspace(0, L, Nx), dtype="float32")
        self.t = tf.convert_to_tensor(np.linspace(0, T, Nt), dtype="float32")
        self.I = I
        self.num_epochs = epochs
        self.g_t_jacobian_func = jacobian(self.g_trial, 0)
        self.g_t_hessian_func = hessian(self.g_trial, 0)
        self.optimizer = optimizers.SGD(learning_rate=0.01)

    def __call__(self):
        pred = self.g_trial(self.model, self.x, self.t)
        return pred

    def get_model(self):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(10, activation="sigmoid", input_shape=(2,)),
                tf.keras.layers.Dense(10, activation="sigmoid"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        return model

    def tf_run(self):
        train_loss_results = []
        model = self.get_model()
        epoch_loss_avg = tf.keras.metrics.Mean()

        for epoch in range(self.num_epochs):
            loss_value, grads = self.grad(model)
            self.optimizer.apply_gradients(zip(grads, model.trainable_variables))
            # Track progress
            epoch_loss_avg.update_state(loss_value)  # Add current batch loss
            train_loss_results.append(epoch_loss_avg.result())
        self.model = model
        return train_loss_results

    def grad(self, model):
        with tf.GradientTape() as tape:
            loss_value = self.cost_function(model)
        t_grad = tape.gradient(loss_value, model.trainable_variables)
        del tape
        return loss_value, t_grad

    def cost_function(self, model):
        """
        Calculate derivatives.
        """
        x, t = tf.convert_to_tensor(self.x, dtype="float32"), tf.convert_to_tensor(
            self.t, dtype="float32"
        )

        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(x)
            tape1.watch(t)
            g_trial = self.g_trial(model, x, t)
            g_x = tape1.gradient(g_trial, x)
            g_xx = tape1.gradient(g_x, x)

        g_t = tape1.gradient(g_trial, t)
        residual = g_xx - g_t
        MSE = tf.reduce_mean(residual ** 2)
        del tape1
        return MSE

    def g_trial(self, model, x, t):
        """
        g_trial(x, t) = h_1(x, t) + h_2(x,t)N(x,t,P)
        h_1 and h_2 are functions to control boundary and inital conditions
        """
        XT = tf.stack([t, x], axis=1)
        h1 = (1 - self.t) * self.I(self.x)
        h2 = self.x * (1 - self.x) * self.t
        return h1 + h2 * model(XT, training=True)


def g_analytic(x, t):
    # Analytic solution to function
    return np.exp(-np.pi ** 2 * t) * I(x)


def I(x):
    # Initial condition
    return np.sin(np.pi * x)


if __name__ == "__main__":
    # Check tensorflow version and eager execution
    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))

    # Set params
    L = 10
    T = 0.5
    dx = 1 / 10
    dt = 5 / 1000  # 0.5*dx**2

    epochs = 100
    ML = PDE_ml_solver(L, T, dx, dt, epochs, I)
    loss = ML.tf_run()

    u = ML()
    x = np.linspace(0, L, int(L / dx))
    plt.plot(x, u[-1])
    plt.show()
    # plt.plot(np.arange(epochs), loss)
    # plt.show()
