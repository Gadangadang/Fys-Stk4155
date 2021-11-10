import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from find_hyperparameters import *
from SGD import *
from NN_functions import *



if __name__ == "__main__":
    digits = datasets.load_digits()
    #plot_digits(digits)

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

    num_hidden_layers = 2
    num_hidden_nodes = 20
    epochs = int(50)
    batch_size = int(35)

    etas = np.logspace(-5, -1, 5)
    lmbds = np.logspace(-5, -1, 5)
    gamma = 0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    X_train, X_test, y_train, y_test = standard_scale(X_train, X_test, y_train, y_test)


    activation = "sigmoid"
    cost_func = "cross_entropy"
    name="mnist"
    best_eta, best_lmbd, best_val = find_hyperparameters(X_train, X_test, y_train, y_test,
                                                         num_hidden_layers,
                                                         num_hidden_nodes,
                                                         batch_size,
                                                         etas,
                                                         lmbds,
                                                         gamma,
                                                         epochs,
                                                         activation,
                                                         cost_func,
                                                         seed=4155,
                                                         name = name,
                                                         return_best = True)
    print(f"{100*best_val:.0f}% NN accuracy. ")
