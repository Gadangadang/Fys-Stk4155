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


    etas = np.logspace(-4, -2.5, 4)
    etasSGDL = np.logspace(-1.5, -0.5, 4)
    lmbds = np.logspace(-4, -2, 4)
    gammas = [0,0.25, 0.5, 0.7]

    activation = "sigmoid"
    cost_func = "cross_entropy"
    loss = "probability"
    test_scores_NN = np.zeros((len(etas), len(lmbds)))
    test_scores_SGDL = np.zeros((len(etas), len(lmbds)))

    for i, eta in enumerate(etas):
        for j, lmbd in enumerate(lmbds):
            print(f"\r(eta_val, lmbd_val) = ({eta},{lmbd})", end="")
            NN = NeuralNetwork(X_train, y_train,
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

            SGDL = SGD(X_train, y_train,
                       etasSGDL[i],
                       m = batche_size,
                       num_epochs = epochs,
                       gamma =
                       gradient_func = "Logistic",
                       loss = "probability", callback = True)

            SGDL.SGD_train()
            NN.train_network_stochastic(int(epochs))
            test_scores_NN[i][j] = NN.get_score(X_test, y_test)
            test_scores_SGDL[i][j] = SGDL.get_score(X_test, y_test)

    plot_heatmap(test_scores_NN, [r"log($\lambda$)",np.log10(lmbds)],[r"log($\eta$)",np.log10(etas)], title = "Grid search Neural Network", name = None)
    plot_heatmap(test_scores_SGDL, [r"Batch sizes",batche_sizes],[r"log($\eta$)",np.log10(etasSGDL)], title = "Grid search Logistic regression", name = None)

    indxNN = np.where(test_scores_NN == np.max(test_scores_NN))
    indxSGDL = np.where(test_scores_SGDL == np.max(test_scores_SGDL))

    print(f"{100*test_scores_NN[indxNN[0]]:.0f}% NN accuracy. ")
    print(f"{100*test_scores_SGDL[indxSGDL[0]]:.0f}% NN accuracy. ")
