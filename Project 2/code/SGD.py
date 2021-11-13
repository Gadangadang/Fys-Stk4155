import numpy as np
import sys
from sklearn.model_selection import train_test_split
from tqdm import trange
import matplotlib.pyplot as plt

class SGD:
    """
    Stochastic Gradient Descent
    with mini batches
    """
    def __init__(self, X, y, eta_val=0.1, m = 0, num_epochs = int(1e4), lmbd = 0 , gamma = 0,  gradient_func = "Ridge", loss = "accuracy", callback = False):
        self.X = X
        self.N = X.shape[0] # Number of data points
        self.y = y
        self.num_categories = y.shape[1]

        dim_check = X.shape[0] == y.shape[0]
        assert dim_check, "Dimensions of X and y does not match"

        self.eta_val = eta_val
        if m == 0: # full gradient descent
            self.m = self.N
        else: # user defined m
            self.m = m

        self.num_epochs = num_epochs
        self.lmbd = lmbd   # Ridge regularization parameter
        self.gamma = gamma  # momentum parameter
        self.vel = 0    # gradient descent "velocity"

        # Set gradient and learning rate functions

        Loss_functions = {"accuracy": self.accuracy_score,
                          "MSE": self.MSE_score,
                          "R2": self.R2_score,
                          "probability": self.probability_score}
        Gradient_funcs = {"Logistic": self.gradient_Logistic, "Ridge": self.gradient_Ridge}

        self.gradient_func = Gradient_funcs[gradient_func]
        self.score_func = Loss_functions[loss]

        self.eta_func = self.eta_const # Default

        self.score_shape = 1
        if loss == "accuracy":
            self.score_shape = self.num_categories

        self.callback_label = loss
        if callback:
            self.callback_print = lambda epoch, score_epoch: print(f"epoch: {epoch}, {self.callback_label} = {score_epoch}")
        else:
             self.callback_print = lambda epoch, score_epoch: None
        # Initializa theta and set epoch = 1
        self.reset()


    def reset(self):
        self.epoch = 0
        self.initialize_theta_normal()


    def initialize_theta_normal(self):
        """
        Initialize theta with normal disttribution
        """
        self.theta = np.random.normal(0, 1, size=(self.X.shape[1], self.num_categories))


    def get_batches(self):
        idx = np.arange(self.N)
        np.random.shuffle(idx)
        int_max = self.N//self.m
        batches = [idx[i*self.m:(i+1)*self.m] for i in range(int_max)]
        if self.N%self.m != 0:
            batches.append(idx[int_max*self.m:])
        return batches


    def SGD_evolve(self):
        """
        Evolve theta with one epoch
        """
        batches = self.get_batches()
        for batch in batches:
            X = self.X[batch]
            y = self.y[batch]
            g = self.gradient_func(X,y)
            self.vel = self.gamma*self.vel + self.eta_func(self.epoch) * g
            self.theta -= self.vel



    def SGD_train(self):
        """
        Run all epochs of SGD
        """

        self.score = np.zeros((self.num_epochs + 1, self.score_shape))
        self.score[self.epoch] = self.get_score(self.X, self.y)
        self.callback_print(self.epoch, self.score[self.epoch])


        while self.epoch < self.num_epochs:
            self.SGD_evolve()
            self.epoch +=1
            self.score[self.epoch] = self.get_score(self.X, self.y)
            self.callback_print(self.epoch, self.score[self.epoch])
        return self.theta.ravel()

    def plot_score_history(self, name=None, legend = []):
        plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
        if self.num_epochs > 150:
            linestyle = "-"
            marker = "None"
        else:
            linestyle = "--"
            marker = "o"
        markersize = 3

        plt.plot(np.linspace(0, self.num_epochs, self.num_epochs),
                 self.score[:self.num_epochs], linestyle=linestyle, marker=marker, markersize=markersize)
        plt.xlabel("epoch", fontsize=14)
        plt.ylabel(self.callback_label, fontsize=14)
        if self.score.shape[1] > 1:
            if len(legend) == 0: # no legend in arg
                plt.legend(["category " + str(i)
                       for i in range(self.score.shape[1])], fontsize=13)
            else: # legend from arg
                plt.legend(legend)
        plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
        if isinstance(name, str):
            plt.savefig(f"../article/figures/{name}_score_history.pdf",
                    bbox_inches="tight")
        plt.show()


    def predict(self, X):
        prediction = np.exp(X @ self.theta)/(1+np.exp(X @ self.theta))
        return prediction

    def get_score(self, X, target):
        return self.score_func(X, target)

    def accuracy_score(self, X, target):
        pred = np.around(self.predict(X))
        hits = np.sum(np.around(pred) == target, axis=0)
        possible = target.shape[0]
        acc = hits / possible
        return acc

    def MSE_score(self, X, target):
        pred = self.predict(X)
        return np.mean((pred - target)**2)

    def R2_score(self, X, target):
        ymod = self.predict(X)
        target = target
        return 1 - np.sum((target - ymod)**2) /\
            np.sum((target - np.mean(target, axis=0) ** 2))

    def probability_score(self, X, target):
        pred = self.predict(X)
        guess = np.argmax(pred, axis=1)
        target = np.argmax(target, axis=1)
        acc = np.sum(guess == target) / len(target)
        return acc

    # --- eta and gradient functions --- #

    def eta_const(self, epoch): # Constant learning rate
        return self.eta_val

    def gradient_Logistic(self, X, y): # Logistic gradient
        g = -X.T @ (y - np.exp(X @ self.theta)/(1+np.exp(X @ self.theta))) + 2*self.lmbd * self.theta
        return g

    def gradient_Ridge(self, X, y): # Ridge gradient
        g = 2*(1/self.N * X.T @ ((X @ self.theta) - y) + self.lmbd*self.theta)
        return g



if __name__ == "__main__":
    # Get modules from project 1
    sys.path.insert(1,"../../Project 1/code/")
    from Functions import *

    # #--- Create data from Franke Function ---#
    # N = 10               # Number of points in each dimension
    # z_noise = 0.2       # Added noise to the z-value
    # n = 3               # Highest order of polynomial for X
    # x, y, z = generate_data(N, z_noise)
    # X = create_X(x, y, n)
    #
    #
    # # --- Test run --- #
    # eta = 0.001
    # m = 0 # m=0 gives full gradient descent
    #
    # #--- Regression ---#
    # solver = SGD(X, z, eta_val=0.1, m = m, num_epochs = int(1e2))
    # solver.gamma = 0.8
    # theta_SGD = solver.SGD_train()      # Stochastic Gradient Descent
    # theta_OLS = OLS_regression(X, z)  # OLS regression
    #
    #
    # ztilde_SGD = (X @ theta_SGD).ravel()
    # ztilde_OLS = (X @ theta_OLS).ravel()
    #
    # print("SGD: ", MSE(ztilde_SGD, z))
    # print("OLS: ", MSE(ztilde_OLS, z))


    # --- Logistic run --- #
    from sklearn import datasets, svm, metrics
    from NN_functions import *
    digits = datasets.load_digits()

    # flatten the images
    data = digits.data
    # Create a classifier: a support vector classifier
    X = data
    y_flat = digits.target
    y = []
    for i in range(len(y_flat)):
        y_i = np.asarray([j == y_flat[i] for j in range(10)])
        y.append(y_i)
    y = np.asarray(y)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_test, y_train, y_test = standard_scale(X_train, X_test, y_train, y_test)


    gamma = 0.5
    seed = 4155
    batch_size = 0
    eta_val = 1e-1
    epochs = 1000




    SGDL = SGD(X_train, y_train, eta_val, m = batch_size, num_epochs = epochs, gradient_func = "Ridge", loss = "probability", callback = True)
    SGDL.SGD_train()
    SGDL.plot_score_history()
    print(SGDL.get_score(X_test, y_test))
