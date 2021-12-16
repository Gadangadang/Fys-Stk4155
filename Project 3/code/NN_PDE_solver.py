import tensorflow as tf
import matplotlib.pyplot as plt
import autograd.numpy as np
from plot_set import *
from autograd import jacobian, hessian, grad
import ExplicitSolver as ES
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.layers import Dense, Input
from tensorflow.data import Dataset
from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_custom_objects
from tqdm import tqdm
from Functions import *


class NeuralNetworkPDE:
    """
    Neural Network class for calculating PDE's. The loss function
    is specialized to solve the 1D diffusion equation on the form
            d_t u(x, t) = A d_xx u(x, t)
    """
    def __init__(self, x, t, epochs, I, lr, in_out = [2, 1]):
        """Initialize class ML object.

        Args:
            x            (ndarrray): x domain discretized as array
            t             (ndarray): t domain discretized as array
            epochs            (int): Number of training epochs
            I                (func): Initial condition function
            lr              (float): learning rate for training
            in_out (list, optional): Structure of the input and output layers of the network
        """

        self.x = tf.cast(tf.convert_to_tensor(x), tf.float32)
        self.t = tf.cast(tf.convert_to_tensor(t), tf.float32)
        self.data = self.create_dataset()

        self.I = I
        self.num_epochs = epochs
        self.learning_rate = lr
        self.in_out = in_out

        self.tracker = self.track_loss
        self.process = []

    def __call__(self):
        """
        Call function for the ML object.

        Returns:
            list: Splits trial function into the solution and the time array
        """
        t, x = self.data[:, 0], self.data[:, 1]
        u = self.g_trial(self.model, x, t).numpy()
        return np.split(u, len(self.t))

    def create_dataset(self):
        xx, tt = tf.meshgrid(self.x, self.t)
        data = tf.stack([tf.reshape(tt, [-1]), tf.reshape(xx, [-1])], axis=1) # NEW


        return data

    def get_model(self):
        """
        Initializes the model, setting up layers and compiles the model,
        prepping for training.

        Returns:
            tensorflow_object: compiled model
        """
        get_custom_objects().update({"abs_activation": self.abs_activation})
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    20, activation="sigmoid", input_shape=(self.in_out[0],)
                ),
                tf.keras.layers.Dense(20, activation="sigmoid"),
                tf.keras.layers.Dense(20, activation="sigmoid"),
                tf.keras.layers.Dense(self.in_out[1]),
            ]
        )
        self.optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=self.optimizer)
        # model.summary() # disabled print temporary
        return model

    def train(self):
        """
        Trains the model and tracks the metric you choose.

        Returns:
            list: list comprising the saved values you choose
        """
        train_loss_results = []
        self.model = self.get_model()
        model = self.model
        try:
            tvals = tqdm(range(self.num_epochs))
            for epoch in tvals:

                # Calculate loss and gradient of loss.
                loss_value, grads = self.grad(model)
                # Update parameters in network.
                self.optimizer.apply_gradients(zip(grads, model.trainable_variables))
                # Track Loss
                self.tracker(loss_value)
                tvals.set_description(self.print_string)
                self.model = model  # Save trained network.
            return self.process
        except KeyboardInterrupt:
            return self.process

    @tf.function
    def grad(self, model):
        """
        Calculates the gradient for the model. Calculates the loss of the model,
        and then calculates a gradient to correct the model.

        Args:
            model (tensorflow_object): the current model

        Returns:
            tensor: the loss value for the given derivation
            tensor: the gradient with respect to the trainable variables
        """
        with tf.GradientTape() as tape:
            tape.watch(model.trainable_variables)
            loss_value = self.cost_function(model)
        t_grad = tape.gradient(loss_value, model.trainable_variables)
        del tape
        return loss_value, t_grad

    @tf.function
    def cost_function(self, model):
        """[summary]

        Args:
            model (tensorflow_object): the current model

        Returns:
            tensor: MSE tensor to use for correction in gradient calculation
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
        del tape1
        del tape2
        residual = g_xx - g_t
        MSE = tf.reduce_mean(tf.square(residual))
        return MSE

    @tf.function
    def g_trial(self, model, x, t):
        """
        Calculates the trial function given either two data points or two arrays

        g_trial(x, t) = h_1(x, t) + h_2(x,t)N(x,t,P)
        h_1 and h_2 are functions to control boundary and inital conditions.


        Args:
            model (tensorflow_object): the current model
            x (tensor): tensor of x data
            t (tensor): tensor of t data

        Returns:
            tensor: returns tensorflow tensor. needs to be converted to numpy to be used
        """
        XT = tf.stack([t, x], axis=1)
        h1 = (1 - t) * self.I(x)
        h2 = x * (1 - x) * t
        return h1 + h2 * tf.squeeze(model(XT, training=True))

    def track_loss(self, loss):
        self.print_string = f"Residual={tf.reduce_mean(loss):.2e}"
        self.process.append(loss)

    def save_model(self, name):
        self.model.save(f"tf_models/model_{name}.h5")

    def load_model(self, name):
        self.model = tf.keras.models.load_model(f"tf_models/model_{name}.h5")

    def save_checkpoint(self, checkpoint_name):
        self.model.save_weights(f"checkpoints/{checkpoint_name}")

    def load_from_checkpoint(self, checkpoint_name):
        self.model.load_weights(f"tf_checkpoints/{checkpoint_name}")

    def abs_activation(self, value):
        # return K.switch(value >= 0, value, -value)
        return tf.math.tanh(value)


def I(x):
    # Initial condition
    return tf.sin(np.pi * x)


if __name__ == "__main__":
    # Check tensorflow version and eager execution
    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))
    tf.random.set_seed(123)
    L = 1
    T = 1
    dx = 0.1
    # dt = 0.5 * dx ** 2
    dt = dx
    lr = 5e-2

    epochs = 2e4
    x = np.linspace(0, L, round(L / dx) + 1)
    t = np.linspace(0, T, round(T / dt) + 1)

    # Place tensors on the CPU
    with tf.device("/CPU:0"):  # Write '/GPU:0' for large networks
        ML = NeuralNetworkPDE(x, t, int(epochs), I, lr)
        loss = ML.train()
        u_complete = ML()


    ESS = ES.ExplicitSolver(I, L, T, dx, dt, 0, 0, False)
    ESS.u_complete = u_complete
    ESS.plot_comparison("Neural Network", name = "NN_PDE_equal", title_extension=f": dx = {dx}")
    loss_plot(loss, name = f"NN_PDE_MSE_NN_PDE_equal_dx_{dx}")

    # ESS.plot_comparison("Neural Network", title_extension=f": dx = {dx}")






    # ML.save_model(f"{epochs:e}epoch_sigmoid")
    # ML.load_model("100000epoch_sigmoid")
    # u_complete = ML()

    # u_complete = np.asarray(u_complete)


    # Run animation against exact solution

    # dt = 0.1 * 0.5 * dx ** 2
    # ESS = ES.ExplicitSolver(I, L, T, dx, dt, 0, 0, False)
    # solution = ESS.run_simulation()
    # ESS.plot_comparison("Explicit solver", title_extension=f": dx = {dx}")

    # ESS.rel_err_plot("Explicit ", t, other_data=u_complete, other_name="NN")

    # Animate
    # ESS.u_complete = u_complete
    # ESS.animator("Neural network", "001_2e3")
    # ESS.plot_comparison("Explicit solver", title_extension=f": dx = {dx}")

    # Save
    # ML.save_model(f"{epochs}epoch")
