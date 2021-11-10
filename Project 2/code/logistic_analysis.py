import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from SGD import *
from NN_functions import *
from NeuralNetwork import NeuralNetwork




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

    gamma = 0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    X_train, X_test, y_train, y_test = standard_scale(X_train, X_test, y_train, y_test)


    num_hidden_layers = 2
    num_hidden_nodes = 30
    batch_size = int(50)
    gamma = 0.9
    seed = 4155
    n_categories = 1
    epochs = int(200)
    batch_size = int(50)


    etas = np.logspace(-4, -1, 3)
    lmbds = np.logspace(-1, 0, 3)


    activation = "sigmoid"
    cost_func = "cross_entropy"
    loss = "probability"
    test_scores = np.zeros((len(etas), len(lmbds)))

    NN = NeuralNetwork(X_train,
                       y_train,
                       num_hidden_layers,
                       num_hidden_nodes,
                       batch_size,
                       etas[0],
                       lmbds[0],
                       gamma,
                       seed,
                       activation,
                       cost="cross_entropy",
                       loss = "probability",
                       callback = False)
    #NN.set_eta_decay( 1, 60)
    NN.train_network_stochastic(int(epochs))
    NN.plot_score_history()
    exit()

    for i, eta in enumerate(etas):
        for j, lmbd in enumerate(lmbds):
            print(f"\r(eta_val, lmbd_val) = ({eta[i]},{lmbds[j]})/({len(etas)-1},{len(lmbds)-1})", end="")
            NN = NeuralNetwork(X_train,
                               y_train,
                               num_hidden_layers,
                               num_hidden_nodes,
                               batch_size,
                               eta,
                               lmbd,
                               gamma,
                               seed,
                               activation,
                               cost_func,
                               loss,
                               callback = True)

            NN.train_network_stochastic(int(epochs), plot = False)
            test_scores[i][j] = NN.get_score(X_test, y_test)

    plot_heatmap(test_scores,[r"log($\eta$)",np.log10(etas)], [r"log($\lambda$)",np.log10(lmbds)], title = None, name = None)
    indx = np.where(test_scores == np.max(test_scores))


    print(f"{100*test_scores[indx]:.0f}% NN accuracy. ")
