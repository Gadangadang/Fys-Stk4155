import os
import sys
import autograd.numpy as np
from autograd import elementwise_grad
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def accuracy_score_numpy(Y_test, Y_pred):
    tol = 1e-2
    return np.sum(np.abs(Y_test - Y_pred) < tol) / len(Y_test)


def test_accuracy_nn():
    import seaborn as sns
    eta_vals = np.logspace(-5, 1, 7)
    lmbd_vals = np.logspace(-5, 1, 7)
    # store the models for later use
    DNN_numpy = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)

    # grid search
    for i, eta in enumerate(eta_vals):
        for j, lmbd in enumerate(lmbd_vals):
            dnn = NeuralNetwork(X_train,
                                Z_train,
                                num_hidden_layers=2,
                                num_hidden_nodes=10,
                                batch_size=batch_size,
                                eta=eta,
                                lmbd=lmbd,
                                seed=4155,
                                activation="sigmoid",
                                cost="MSE")

            dnn.run_network_stochastic(epochs)

            DNN_numpy[i][j] = dnn

            test_predict = dnn.predict(X_test)
            for _ in range(test_predict.shape[0]):
                print(Z_test[_], " ", test_predict[_])

            print("Learning rate  = ", eta)
            print("Lambda = ", lmbd)
            #print(np.shape(Z_test), np.shape(test_predict))
            print("Accuracy score on test set: ",
                  accuracy_score_numpy(Z_test, test_predict))
            print()

    sns.set()

    train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
    test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))

    for i in range(len(eta_vals)):
        for j in range(len(lmbd_vals)):
            dnn = DNN_numpy[i][j]

            train_pred = dnn.predict(X_train)
            test_pred = dnn.predict(X_test)

            train_accuracy[i][j] = accuracy_score_numpy(Z_train, train_pred)
            test_accuracy[i][j] = accuracy_score_numpy(Z_test, test_pred)

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(train_accuracy, annot=True, ax=ax, cmap="viridis")
    ax.set_title("Training Accuracy")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
    ax.set_title("Test Accuracy")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.show()

if __name__ == "__main__":
    # Get modules from project 1
    path = os.getcwd()  # Current working directory
    path += '/../../Project 1/code'
    sys.path.append(path)
    from Functions import *
    from NeuralNetwork import *
    # The above imports numpy as np so we have to redefine:
    import autograd.numpy as np
    #--- Create data from Franke Function ---#
    N = 10               # Number of points in each dimension
    z_noise = 0.2       # Added noise to the z-value
    n = 8               # Highest order of polynomial for X
    epochs = 10000
    iterations = 1
    batch_size = int(N * N * 0.8)

    x, y, z = generate_data(N, z_noise)
    X = create_X(x, y, n)

    X_train, X_test, Z_train, Z_test = train_test_split(X, z, test_size=0.2)
    
    test_accuracy_nn()  # Test accuracy of network
