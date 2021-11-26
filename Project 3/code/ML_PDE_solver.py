import tensorflow as tf
import matplotlib.pyplot as plt
import autograd.numpy as np
from plot_set import *
from autograd import jacobian, hessian, grad

from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.layers import Dense, Input
from tensorflow.data import Dataset
from tqdm import tqdm


class PDE_ml_solver:
    def __init__(self, L, T, dx, dt, epochs, I, batch_size):
        self.x = tf.cast(tf.linspace(0, L, int(L / dx)), tf.float32)
        self.t = tf.cast(tf.linspace(0., T, int(T / dt)), tf.float32)
        self.dataset = self.create_dataset(self.x,self.t,batch_size)
        self.I = I
        self.num_epochs = epochs
        self.g_t_jacobian_func = jacobian(self.g_trial, 0)
        self.g_t_hessian_func = hessian(self.g_trial, 0)
        self.optimizer = optimizers.SGD(learning_rate=0.01)

    def __call__(self):
        sol = []
        #pred = self.g_trial(self.model, self.x, self.t)
        for i, ti in enumerate(self.t):
            sol.append([])
            for xi in self.x:
                sol[i].append(self.g_trial(self.model, tf.Variable([xi]), tf.Variable([ti]))[0][0].numpy())

        return sol

    def create_dataset(self,x,t,batch_size):
        data = tf.stack([tf.random.shuffle(t),tf.random.shuffle(x)], axis=1)#Dataset.zip((t, x))
        data = Dataset.from_tensor_slices(data)
        data = data.batch(batch_size)
        return data

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

        for epoch in tqdm(range(self.num_epochs)):
            loss_epoch = 0
            for step, batch in enumerate(self.dataset):
                self.batch = batch
                loss_value, grads = self.grad(model)
                self.optimizer.apply_gradients(zip(grads, model.trainable_variables))
                # Track progress
                #epoch_loss_avg.update_state(loss_value)  # Add current batch loss
                #train_loss_results.append(epoch_loss_avg.result())
                train_loss_results.append(tf.math.reduce_mean(loss_value))
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
        t,x = self.batch[:,0], self.batch[:,1]
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(x)

            with tf.GradientTape() as tape2:
                tape2.watch(t)
                g_trial = self.g_trial(model, x, t)
                g_x = tape1.gradient(g_trial, x)

        g_xx = tape1.gradient(g_x, x)
        g_t = tape2.gradient(g_trial, t)
        del tape1
        del tape2
        residual = (g_xx - g_t)
        #MSE = residual
        return tf.math.abs(residual)

    def g_trial(self, model, x, t):
        """
        g_trial(x, t) = h_1(x, t) + h_2(x,t)N(x,t,P)
        h_1 and h_2 are functions to control boundary and inital conditions
        """
        XT = tf.stack([t, x], axis=1)
        h1 = (1 - t) * self.I(x)
        h2 = x * (1 - x) * t
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
    dt = 0.005  # 0.5*dx**2


    epochs = 500
    ML = PDE_ml_solver(L, T, dx, dt, epochs, I, 50)
    loss = ML.tf_run()

    u = ML()
    x = np.linspace(0, L, int(L / dx))
    t = np.linspace(0, T, int(T / dt))
    for i in range(0,100,10):
        plt.plot(x, u[i])
        plt.plot(x,g_analytic(x,dt*i), "--")


    #plt.plot(x,g_analytic(x,dt), "--")
    #plt.xlim([0,1])
    plt.show()
    """plt.plot(np.arange(len(loss)), loss)
    plt.show()
    """
