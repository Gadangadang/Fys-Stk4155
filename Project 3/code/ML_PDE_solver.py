import tensorflow as tf
import matplotlib.pyplot as plt
import autograd.numpy as np
from plot_set import *
from autograd import jacobian, hessian, grad
import ExplicitSolver as ES
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.layers import Dense, Input
from tensorflow.data import Dataset
from tqdm import tqdm


class NeuralNetworkPDE:
    def __init__(self, x, t, epochs, I, lr):
        self.x = tf.cast(tf.convert_to_tensor(x), tf.float32)
        self.t = tf.cast(tf.convert_to_tensor(t), tf.float32)
        self.data = self.create_dataset()

        self.I = I
        self.num_epochs = epochs
        self.learning_rate = lr

        self.g_t_jacobian_func = jacobian(self.g_trial, 0)
        self.g_t_hessian_func = hessian(self.g_trial, 0)

    def __call__(self):
        t, x = self.data[:, 0], self.data[:, 1]
        u = self.g_trial(self.model, t, x).numpy()
        return np.split(u, len(self.t))

    def create_dataset(self):
        T, X = tf.meshgrid(self.t, self.x)
        data = tf.stack([tf.reshape(T, [-1]), tf.reshape(X, [-1])], axis=1)
        return data

    def get_model(self):
        model = tf.keras.Sequential(
               [tf.keras.layers.Dense(20, activation="sigmoid", input_shape=(2,)),
                tf.keras.layers.Dense(20, activation="sigmoid"),
                tf.keras.layers.Dense(20, activation="sigmoid"),
                tf.keras.layers.Dense(1),])
        self.optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=self.optimizer)
        model.summary()
        return model

    def train(self):
        train_loss_results = []
        model = self.get_model()
        try:
            tvals = tqdm(range(self.num_epochs))
            for epoch in tvals:
                # Calculate loss and gradient of loss.
                loss_value, grads = self.grad(model)
                # Update parameters in network.
                self.optimizer.apply_gradients(
                    zip(grads, model.trainable_variables))
                # Track Loss
                train_loss_results.append(loss_value)
                tvals.set_description(f"Residual={loss_value:.3f}")
            self.model = model  # Save trained network.
            return train_loss_results
        except:
            self.model = model  # Save trained network.
            return train_loss_results

    @tf.function
    def grad(self, model):
        with tf.GradientTape() as tape:
            tape.watch(model.trainable_variables)
            loss_value = self.cost_function(model)
        t_grad = tape.gradient(loss_value, model.trainable_variables)
        del tape
        return loss_value, t_grad


    @tf.function
    def cost_function(self, model):
        """
        Calculate derivatives.
        """
        with tf.GradientTape() as tape1:
            t, x = self.data[:, 0], self.data[:, 1]
            tape1.watch(x)
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch([x, t])
                g_trial = self.g_trial(model, x, t)
            g_x = tape2.gradient(g_trial, x)
            g_t = tape2.gradient(g_trial, t)

        g_xx = tape1.gradient(g_x, x)
        del tape1; del tape2
        residual = g_xx - g_t
        MSE = tf.reduce_mean(tf.square(residual))
        return MSE

    @tf.function
    def g_trial(self, model, x, t):
        """
        g_trial(x, t) = h_1(x, t) + h_2(x,t)N(x,t,P)
        h_1 and h_2 are functions to control boundary and inital conditions.
        """
        XT = tf.stack([t, x], axis=1)
        h1 = (1 - t) * self.I(x)
        h2 = x * (1 - x) * t
        return h1 + h2 * tf.squeeze(model(XT, training=True))


def g_analytic(x, t):
    # Analytic solution to function
    return np.exp(-np.pi ** 2 * t) * I(x)


def I(x):
    # Initial condition
    return tf.sin(np.pi * x)


if __name__ == "__main__":
    # Check tensorflow version and eager execution
    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))

    #v1 = np.arange(6).reshape((1,6))
    #v2 = v1.reshape((6,1))


    #print(v2@v1)
    #exit()

    L = 1
    T = 1
    dx = 0.01
    dt = 0.01
    lr = 5e-2

    epochs = 1000
    x = np.linspace(0, L, int(L / dx))
    t = np.linspace(0, T, int(T / dt))
    ML = NeuralNetworkPDE(x, t, epochs, I, lr)
    loss = ML.train()


    plt.plot(np.arange(len(loss)), loss, label="Loss")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.title("MSE as function of epochs")
    plt.show()

    # Run animation against exact solution
    ESS = ES.ExplicitSolver(I, L, T, dx, dt, 0, 0)
    u_complete = ML()
    ESS.u_complete = u_complete
    ESS.animator("Neural network")
    ESS.plot_comparison("Neural network")
