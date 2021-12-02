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



class bias_variance():
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test


    def set_method(self, method, complexity_array):
        methods = {"decision_tree": self.decision_tree_model, "boosting": self.boosting_model}
        complexity_labels = {"decision_tree": "tree depth", "boosting": "number of estimators"}
        self.complexity_label = complexity_labels[method]
        self.method = methods[method]
        self.n_arr = complexity_array
        self.bias = np.zeros(len(self.n_arr))
        self.variance = np.zeros(len(self.n_arr))
        self.MSE_test = np.zeros(len(self.n_arr))


    def decision_tree_model(self, complexity):
        return DecisionTreeRegressor(max_depth=complexity)

    def boosting_model(self, complexity):
        return AdaBoostRegressor(random_state=0, n_estimators=complexity)


    def feed_forward_network():
        pass

    def bootstrap(self, num_bootstraps):
        y_pred = np.empty((y_test.shape[0], num_bootstraps))
        for b in range(num_bootstraps):
            X_, y_ = resample(self.X_train, self.y_train)
            self.model.fit(X_, y_.ravel())
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
