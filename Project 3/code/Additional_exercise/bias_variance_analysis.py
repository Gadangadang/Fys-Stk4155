from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

from sklearn.utils import resample
from data import *
from tqdm import tqdm

import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from plot_set import *

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
import tensorflow as tf




class bias_variance():
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test


    def set_method(self, method, complexity_array, *args):
        methods =   {   "decision_tree":      [self.decision_tree_model   ,"tree depth"],
                        "boosting":           [self.boosting_model        ,"# estimators"],
                        "NN_variable_nodes":  [self.NN_model_nodes             ,"# hidden nodes"],
                        "NN_variable_layers": [self.NN_model_layers              ,"# hidden layers"]}

        self.method, self.complexity_label = methods[method]
        self.n_arr = complexity_array
        self.bias = np.zeros(len(self.n_arr))
        self.variance = np.zeros(len(self.n_arr))
        self.MSE_test = np.zeros(len(self.n_arr))
        self.args = args
        self.fit = lambda X, y: self.model.fit(X, y)




    def decision_tree_model(self, complexity):
        return DecisionTreeRegressor(max_depth=complexity)

    def boosting_model(self, complexity):
        return AdaBoostRegressor(random_state=0, n_estimators=complexity)

    def NN_model_nodes(self, complexity):
        num_hidden_nodes = complexity
        num_hidden_layers = self.args[0]
        return self.feed_forward_network(num_hidden_nodes, num_hidden_layers)


    def NN_model_layers(self, complexity):
        num_hidden_nodes = self.args[0]
        num_hidden_layers = complexity
        return self.feed_forward_network(num_hidden_nodes, num_hidden_layers)


    def feed_forward_network(self, num_hidden_nodes, num_hidden_layers):
        epochs = 12
        model = Sequential()
        model.add(Input(shape=self.X_train.shape[1]))
        for i in range(num_hidden_layers):
            model.add(Dense(num_hidden_nodes))
        model.add(Dense(1))
        model.compile(optimizer='sgd', loss='mse')
        self.fit = lambda X, y: self.model.fit(X, y, epochs = epochs ,verbose = 0)
        return model




    def bootstrap(self, num_bootstraps):
        y_pred = np.empty((y_test.shape[0], num_bootstraps))
        for b in range(num_bootstraps):
            X_, y_ = resample(self.X_train, self.y_train)
            self.fit(X_, y_.ravel())
            y_pred[:, b] = self.model.predict(X_test).ravel()
        return y_pred

    def analyze(self, num_bootstraps = 200):
        for i, n in enumerate(tqdm(self.n_arr)):
            self.model = self.method(n)
            y_pred = self.bootstrap(num_bootstraps)
            self.bias[i] = np.mean((self.y_test - np.mean(y_pred, axis=1, keepdims=True))**2)
            self.variance[i] = np.mean(np.var(y_pred, axis=1))
            self.MSE_test[i] = np.mean(np.mean((self.y_test - y_pred)**2, axis=1, keepdims=True))


    def plot(self):
        plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
        plt.plot(self.n_arr, self.bias, "o-", label= r"Bias$^2$")
        plt.plot(self.n_arr, self.variance, "o-", label= "Variance")
        plt.plot(self.n_arr, self.MSE_test, "o-", label= "MSE test")

        plt.xlabel(self.complexity_label, fontsize=14)
        plt.ylabel(r"Error", fontsize=14)
        plt.legend(fontsize = 13)
        plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
        plt.show()
        # plt.savefig("../article/figures/figure.pdf", bbox_inches="tight")




if __name__ == "__main__":
    X_train, X_test, y_train, y_test = fetch_data()
    n_arr = np.arange(1, 15)

    bv_object = bias_variance(X_train, X_test, y_train, y_test)
    bv_object.set_method("decision_tree", n_arr)
    bv_object.analyze()
    bv_object.plot()

    bv_object.set_method("boosting", n_arr)
    bv_object.analyze()
    bv_object.plot()

    n_arr = np.linspace(1, 100, 10).astype("int")
    bv_object.set_method("NN_variable_nodes", n_arr, 2)
    bv_object.analyze(10)
    bv_object.plot()

    n_arr = np.linspace(1, 200, 10).astype("int")
    bv_object.set_method("NN_variable_layers", n_arr, 20)
    bv_object.analyze(10)
    bv_object.plot()
