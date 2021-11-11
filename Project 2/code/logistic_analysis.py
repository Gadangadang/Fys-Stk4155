from sklearn import datasets, svm, metrics
from SGD import *
from NN_functions import *
from NeuralNetwork import NeuralNetwork
from sklearn.linear_model import LinearRegression




if __name__ == "__main__":
    digits = datasets.load_digits()

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

    X_train, X_test = standard_scale(X_train, X_test)

    num_hidden_layers = 2
    num_hidden_nodes = 30
    batch_size = int(50)
    gamma = 0.9
    seed = 4155
    n_categories = 1
    epochs = int(200)
    batch_size = int(50)


    etas = np.logspace(-4, -2, 4)
    etasSGDL = np.logspace(-4, -3, 4)
    lmbds = np.logspace(-3, -1, 4)
    lmbdsSGDL = np.logspace(-1, 0.5, 4)
    #gammas = np.linspace(0, 0.5, 4)

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
                       m = batch_size,
                       num_epochs = epochs,
                       lmbd = lmbd*100,
                       gradient_func = "Logistic",
                       loss = "probability", callback = True)

            SGDL.SGD_train()
            NN.train_network_stochastic(int(epochs))
            test_scores_NN[i][j] = NN.get_score(X_test, y_test)
            test_scores_SGDL[i][j] = SGDL.get_score(X_test, y_test)

    plot_heatmap(test_scores_NN, [r"log($\lambda$)",np.log10(lmbds)],[r"log($\eta$)",np.log10(etas)], title = "Grid search Neural Network", name = None)
    plot_heatmap(test_scores_SGDL, [r"log($\lambda$)",np.log10(lmbds)],[r"log($\eta$)",np.log10(etasSGDL)], title = "Grid search Logistic regression", name = None)

    #Scikit learn
    reg = LinearRegression().fit(X_train, y_train)
    pred = reg.predict(X_test)
    guess = np.argmax(pred, axis=1)
    target = np.argmax(y_test, axis=1)
    Scikit_best = np.sum(guess == target) / len(target)

    indxNN = np.where(test_scores_NN == np.max(test_scores_NN))
    indxSGDL = np.where(test_scores_SGDL == np.max(test_scores_SGDL))
    NN_best = test_scores_NN[indxNN[0]][0][0]
    SGDL_best = test_scores_SGDL[indxNN[0]][0][0]

    print(f"{100*NN_best:.0f}% NN accuracy. ")
    print(f"{100*SGDL_best:.0f}% Logistic regression accuracy. ")
    print(f"{100*Scikit_best:.0f}% Sklearn logistic regression accuracy. ")

    NN = NeuralNetwork(X_train, y_train,
                       num_hidden_layers,
                       num_hidden_nodes,
                       batch_size,
                       etas[indxNN[0][0]],
                       lmbds[indxNN[0][1]],
                       gamma,
                       seed,
                       activation,
                       cost_func,
                       loss,
                       callback = True)

    NN.train_network_stochastic(int(epochs))
    NN.layers_a[0] = X_test
    NN.feed_forward()
    predict = NN.soft_max_activation(NN.layers_z[-1])
    predict = np.argmax(predict, axis=1)
    target = np.argmax(y_test, axis=1)

    plot_predictions(X_test, predict, target)
