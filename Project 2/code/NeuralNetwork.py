import os
import sys
import autograd.numpy as np
from autograd import elementwise_grad
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


class NeuralNetwork:
    """[summary]
    """

    def __init__(self,
                 X,
                 t,
                 num_hidden_layers=1,
                 num_hidden_nodes=10,
                 batch_size=4,
                 eta=0.001,
                 lmbd=0.00,
                 gamma = 0.0,
                 seed=4155,
                 activation="sigmoid",
                 cost="MSE",
                 callback = "None"):
        """[summary]

        Args:
            X ([type]): [description]
            t ([type]): [description]
            num_hidden_layers (int, optional): [description]. Defaults to 2.
            num_hidden_nodes (int, optional): [description]. Defaults to 10.
            batch_size (int, optional): [description]. Defaults to 1.
            eta (float, optional): [description]. Defaults to 0.001.
            lmbd (float, optional): [description]. Defaults to 0.0.
            seed (int, optional): [description]. Defaults to 4155.
            activation (str, optional): [description]. Defaults to "sigmoid".
            cost (str, optional): [description]. Defaults to "MSE".
        """
        self.X = X  # Design matrix shape: N x features --> features x N
        self.t = t  # Target, shape: categories x N
        self.T = np.copy(t)

        self.N = X.shape[0]
        self.data_indices = np.arange(self.N)
        self.num_features = X.shape[1]
        self.num_categories = t.shape[1]

        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_nodes = num_hidden_nodes
        self.num_output_nodes = self.num_categories
        self.L = self.num_hidden_layers + 2  # number of layer in total

        self.batch_size = batch_size
        self.eta = eta
        self.lmbd = lmbd
        self.gamma = gamma
        self.seed = seed

        self.create_layers()
        self.create_biases_and_weights()

        if activation == "sigmoid":
            self.activation = self.sigmoid_activation
        elif activation == "relu":
            self.activation = self.RELU_activation
        elif activation == "leaky_relu":
            self.activation = self.Leaky_RELU_activation
        elif activation == "soft_max":
            self.activation = self.soft_max_activation
        if cost == "MSE":
            self.cost = self.MSE

        elif cost == "cross_entropy":
            self.cost = self.cross_entropy
        self.activation_der = elementwise_grad(self.activation)
        self.cost_der = elementwise_grad(self.cost)

        if callback == "accuracy":
            self.callback = True
            self.callback_func = self.accuracy_callback
        elif callback == "MSE":
            self.callback = True
            self.callback_func = self.MSE_callback
        elif callback == "R2":
            self.callback = True
            self.callback_func = self.R2_callback
        else:
            self.callback = False

        self.callback_label = callback



    def callback_print(self, epoch, score_epoch):
        print(f"epoch: {epoch}, {self.callback_label} = {score_epoch}")



    def create_layers(self):
        """
        layers_a: contain activation values of all nodes
        layers_z: contain weighted sum z / unactivated values of all nodes
        """
        self.layers_a = [np.zeros((self.N, self.num_hidden_nodes), dtype=np.float64)
                         for i in range(self.num_hidden_layers)]  # Intialized with the hidden layers

        self.layers_a.insert(0, self.X.copy())  # Add input layer
        self.layers_a.append(
            np.zeros((np.shape(self.t)), dtype=np.float64))  # Add output layer
        self.layers_z = self.layers_a.copy()

    def create_biases_and_weights(self):
        """[summary]
        """
        np.random.seed(self.seed)
        num_hidden_layers = self.num_hidden_layers
        num_features = self.num_features
        num_hidden_nodes = self.num_hidden_nodes
        num_categories = self.num_categories
        bias_shift = 0.1

        self.weights = [np.random.randn(
            num_hidden_nodes, num_hidden_nodes) for i in range(self.num_hidden_layers - 1)]
        self.weights.insert(0, np.random.randn(num_hidden_nodes, num_features))

        # insert unused weight to get nice indexes
        self.weights.insert(0, np.nan)
        self.weights.append(np.random.randn(
            self.num_output_nodes, num_hidden_nodes))

        self.bias = [np.ones(num_hidden_nodes) * bias_shift
                     for i in range(self.num_hidden_layers)]
        self.bias.insert(0, np.nan)  # insert unused bias to get nice indexes
        self.bias.append(np.ones(num_categories) * bias_shift)



        # velocity for momentum
        self.vel_weights = [np.nan]
        self.vel_bias = [np.nan]
        for i in range(1, num_hidden_layers+2):
            self.vel_weights.append(np.zeros(np.shape(self.weights[i])))
            self.vel_bias.append(np.zeros(np.shape(self.bias[i])))


        # local gradient
        self.local_gradient = self.layers_a.copy()  # also called error
        self.local_gradient[0] = np.nan  # don't use first


    def update_parameters(self):
        """[summary]
        """
        self.backpropagation()
        for l in range(1, self.L):
            self.vel_weights[l] = self.gamma*self.vel_weights[l] + self.eta*(self.local_gradient[l].T @ self.layers_a[l - 1] + self.weights[l]*self.lmbd)
            self.weights[l] -= self.vel_weights[l]

            self.vel_bias[l] = self.gamma*self.vel_bias[l] +  self.eta*np.mean(self.local_gradient[l], axis=0)
            self.bias[l] -= self.vel_bias[l]


    def feed_forward(self):
        """[summary]
        """
        for l in range(1, self.L):
            Z_l = self.layers_a[l - 1] @ self.weights[l].T + \
                self.bias[l][np.newaxis, :]
            self.layers_z[l] = Z_l
            self.layers_a[l] = self.activation(Z_l)

    def backpropagation(self):
        """
        Returns the gradient of the cost function
        """

        self.local_gradient[-1] = self.cost_der(
            self.layers_a[-1]) * self.activation_der(self.layers_z[-1])

        for l in reversed(range(1, self.L - 1)):
            self.local_gradient[l] = self.local_gradient[l + 1]\
                @ self.weights[l + 1] * self.activation_der(self.layers_z[l])

    def predict(self, X):
        """
        [summary]

        Args:
            X ([type]): [description]

        Returns:
            [type]: [description]
        """
        self.layers_a[0] = X
        self.feed_forward()
        return self.layers_a[-1]

    def train_network_stochastic(self, epochs, plot = False):
        """[summary]

        Args:
            epochs ([type]): [description]
        """

        if not (self.callback):
            for _ in range(epochs):
                batches = self.get_batches()
                for batch in batches:
                    self.choose_mini_batch(batch)
                    self.feed_forward()
                    self.update_parameters()
        else:
            score = np.zeros((epochs+1, self.num_categories))
            for epoch in range(epochs):
                batches = self.get_batches()
                score[epoch] = self.callback_func()
                self.callback_print(epoch, score[epoch])
                for batch in batches:
                    self.choose_mini_batch(batch)
                    self.feed_forward()
                    self.update_parameters()
            epoch += 1
            score[epoch] = self.callback_func()
            self.callback_print(epoch, score[epoch])

            if plot:
                plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
                plt.plot(np.linspace(0,epochs, epochs+1), score, "o-")
                plt.xlabel(r"$epoch$", fontsize=14)
                plt.ylabel(self.callback_label, fontsize=14)
                plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
                # plt.savefig("../article/figures/figure.pdf", bbox_inches="tight")
                plt.show()

#
#
#

    def MSE_callback(self):
        t_model = self.predict(self.X)
        return np.mean((self.T.ravel() - t_model.ravel())**2)

    # def R2_callback(self):
    #     t_model = self.predict(self.X)
    #     return 1-np.sum((self.T.ravel() - t_model.ravel())**2) / np.sum((self.T.ravel() - np.mean(self.T.ravel())) ** 2)

    def accuracy_callback(self): # write this
        return self.accuracy_score(self.X, self.T)



    def get_batches(self):
        idx = np.arange(self.N)
        np.random.shuffle(idx)
        int_max = self.N//self.batch_size
        batches = [idx[i*self.batch_size:(i+1)*self.batch_size] for i in range(int_max)]
        if self.N % self.batch_size != 0:
            batches.append(idx[int_max*self.batch_size:])
        return batches

    def choose_mini_batch(self, batch):
        """[summary]"""
        self.layers_a[0] = self.X[batch]
        self.t = self.T[batch]

    """
    Activation funtions
    """

    def sigmoid_activation(self, value):
        """[summary]

        Args:
            value ([type]): [description]

        Returns:
            [type]: [description]
        """
        return 1.0 / (1.0 + np.exp(-value))

    def sigmoid_activation_man_der(self, value):
        """[summary]

        Args:
            value ([type]): [description]

        Returns:
            [type]: [description]
        """
        sig = self.sigmoid_activation(value)
        return sig * (sig - 1)

    def RELU_activation(self, value):
        """[summary]

        Args:
            value ([type]): [description]

        Returns:
            [type]: [description]
        """
        vals = np.where(value > 0, value, 0)
        return vals

    def Leaky_RELU_activation(self, value):
        """[summary]

        Args:
            value ([type]): [description]

        Returns:
            [type]: [description]
        """
        vals = np.where(value > 0, value, 0.01 * value)
        return vals

    def soft_max_activation(self, value):
        """[summary]

        Args:
            value ([type]): [description]

        Returns:
            [type]: [description]
        """
        val_exp = np.exp(value)
        return val_exp / (np.sum(val_exp, axis=1, keepdims=True))

    def accuracy_score(self, X, target):
        """[summary]"""
        self.layers_a[0] = X
        self.feed_forward()

        pred = np.around(self.predict(self.X))
        hits = np.sum(np.around(pred) == target, axis = 0)
        possible = target.shape[0]
        acc = hits/possible

        return acc

    def softmax_score(self, X, target):
        pass

        # pred = self.soft_max_activation(self.layers_z[-1])
        # guess = np.argmax(pred, axis=1)
        # target = np.argmax(target, axis=1)
        # val = np.sum(guess == target)/len(target)

    """
    Cost funtions
    """

    def MSE(self, y_tilde):
        """[summary]

        Args:
            y_tilde ([type]): [description]

        Returns:
            [type]: [description]
        """
        return (y_tilde - self.t)**2



    def cross_entropy(self, y_tilde):
        """[summary]

        Args:
            y_tilde ([type]): [description]

        Returns:
            [type]: [description]
        """
        return -(self.t * np.log(y_tilde) + (1 - self.t) * np.log(1 - y_tilde))

    def __str__(self):
        text = "Information of the Neural Network \n"
        text += "Hidden layers:      {} \n".format(self.num_hidden_layer)
        text += "Hidden nodes:       {} \n".format(self.num_hidden_nodes)
        text += "Output nodes:       {} \n".format(self.num_output_nodes)
        text += "Number of features: {} \n".format(self.X.shape[1])

        return text




if __name__ == "__main__":
    # Get modules from project 1
    path = os.getcwd()  # Current working directory
    path += '/../../Project 1/code'
    sys.path.append(path)
    from Functions import *
    # The above imports numpy as np so we have to redefine:
    import autograd.numpy as np
    #--- Create data from Franke Function ---#
    N = 5               # Number of points in each dimension
    z_noise = 0       # Added noise to the z-value
    n = 3               # Highest order of polynomial for X
    epochs = 50

    batch_size = int(N * N * 0.8)
    x, y, z = generate_data(N, z_noise)
    X = create_X(x, y, n)
    X_train, X_test, Z_train, Z_test = train_test_split(X, z, test_size=0.2)
    beta = OLS_regression(X_train, Z_train)
    z_ols = X_test @ beta
    MM = NeuralNetwork(X_train,
                       Z_train,
                       num_hidden_layers=4,
                       num_hidden_nodes=5,
                       batch_size=batch_size,
                       eta=0.001,
                       lmbd=0.0,
                       seed=4155,
                       activation="sigmoid",
                       cost="MSE")
    MM.train_network_stochastic(epochs, plot = True)
    print("Neural Network stochastic", MSE(Z_test, MM.predict(X_test)))
    print("           OLS           ", MSE(Z_test, z_ols))
