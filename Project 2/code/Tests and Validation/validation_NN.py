# This allows using whichever regularizer we want (l1,l2,l1_l2)
from matplotlib.ticker import MaxNLocator
from import_folders import *
import_all_folders()

from NN_functions import plot_heatmap
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import regularizers
# This allows using whichever optimiser we want (sgd,adam,RMSprop)
from tensorflow.keras import optimizers
# This allows defining the characteristics of a particular layer
from tensorflow.keras.layers import Dense
# This allows appending layers to existing models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
#from plot_set import *

from NeuralNetwork import NeuralNetwork


# sklearn classifier

# Tensorflow


def get_logic_gates():
    # Input
    x1 = np.array([0, 0, 1, 1])
    x2 = np.array([0, 1, 0, 1])
    X = np.array([x1, x2]).T  # Design matrix

    # Output
    y_AND = np.array([0, 0, 0, 1])  # And
    y_OR = np.array([0, 1, 1, 1])  # Or
    y_XOR = np.array([0, 1, 1, 0])  # Exclusive OR
    y_gates = np.array([y_AND, y_OR, y_XOR]).T

    return X, y_gates


def logic_gates_NN():
    X, y_gates = get_logic_gates()

    # NN architecture
    num_hidden_layers = 1
    num_hidden_nodes = 4
    batch_size = 4  # Full GD
    eta = 0.9
    lmbd = 0
    gamma = 0

    activation = "sigmoid"
    cost_func = "cross_entropy"
    epochs = 90

    NN = NeuralNetwork(X, y_gates,
                       num_hidden_layers,
                       num_hidden_nodes,
                       batch_size,
                       eta,
                       lmbd,
                       gamma,
                       seed=4155,
                       activation=activation,
                       cost=cost_func,
                       loss="accuracy",
                       callback=True)

    NN.train_network_stochastic(epochs)
    NN.plot_score_history(name="logic_gates", legend=["AND", "OR", "XOR"])
    final_accuracy = NN.get_score(NN.X, NN.T)


def logic_gates_OLS():
    X, y_gates = get_logic_gates()
    gates = ["AND", "OR", "XOR"]
    for i in range(y_gates.shape[1]):
        y = y_gates[:, i].reshape(4, 1)
        theta_OLS = OLS_regression(X, y)
        pred = np.around(X @ theta_OLS)
        print(f"{gates[i]}-gate prediction: {pred}")

    plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
    # plt.figure(num=0, figsize = (3,4), facecolor='w', edgecolor='k')

    plt.plot([X[0, 0], X[3, 0]], [X[0, 1], X[3, 1]],
             "o", markersize=10, label="y = 0")
    plt.plot(X[1:3, 0], X[1:3, 1], "o", marker="P",
             markersize=10, label="y = 1")
    plt.xticks([0, 1])
    plt.yticks([0, 1])

    plt.xlabel(r"$x_1$", fontsize=14)
    plt.ylabel(r"$x_2$", fontsize=14)
    plt.legend(loc="center", fontsize=13)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    plt.savefig("../article/figures/XOR_graphic.pdf", bbox_inches="tight")
    plt.show()


def Franke_NN():
    print("work here sakki")

    np.random.seed(0)
    noise = 0.2
    N = 100
    x = np.random.uniform(0, 1, N)
    y = np.random.uniform(0, 1, N)

    X = np.array([x, y]).T

    # exit()
    z = FrankeFunction(x, y) + noise * np.random.randn(N)
    z = z.reshape(N, 1)

    X_train, X_test, Z_train, Z_test = train_test_split(X, z, test_size=0.2)
    X_train, X_test = mean_scale(X_train, X_test)

    print(X_train.shape, Z_train.shape)

    eta = 0.001
    epochs = 100
    gamma = 0
    lmbd = 0.0
    lay = 2g
    nodes = 50
    batch_size = 15

    num_hidden_layers = 2
    num_hidden_nodes = 70
    batch_size = int(25)
    gamma = 0
    seed = 4155
    n_categories = 1
    epochs = int(400)

    etas = np.logspace(-4, -1, 6)
    lmbds = np.logspace(-4, -1, 6)

    activation = "sigmoid"
    cost_func = "MSE"
    loss = "MSE"
    test_scores_NN = np.zeros((len(etas), len(lmbds)))

    # eta and lambda

    for i, eta in enumerate(etas):
        for j, lmbd in enumerate(lmbds):
            print(f"\r(eta_val, lmbd_val) = ({eta},{lmbd})", end="")
            NN = NeuralNetwork(X_train, Z_train,
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
                               callback=False)

            NN.train_network_stochastic(int(epochs))
            test_scores_NN[i][j] = NN.get_score(X_test, Z_test)

    plot_heatmap(test_scores_NN, [r"log($\lambda$)",np.log10(lmbds)],
                  [r"log($\eta$)",np.log10(etas)], title = "Grid search Neural Network", name = None)

    #layers and mse

    lmbd = 10**(-4)
    eta = 0.1

    num_hidden_layers = np.arange(1,20)

    num_nodes = np.arange(1,10)
    for node in num_nodes:
        test_scores_train = np.zeros(len(num_hidden_layers))
        test_scores_test = np.zeros(len(num_hidden_layers))
        for i, num in enumerate(num_hidden_layers):
            NN = NeuralNetwork(X_train, Z_train,
                               num,
                               node,
                               batch_size,
                               eta,
                               lmbd,
                               gamma,
                               seed,
                               activation,
                               cost_func,
                               loss,
                               callback=False)

            NN.train_network_stochastic(int(epochs))
            test_scores_train[i] = NN.get_score(X_train, Z_train)
            test_scores_test[i] = NN.get_score(X_test, Z_test)

        plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.plot(num_hidden_layers, test_scores_train, label="Train MSE")
        plt.plot(num_hidden_layers, test_scores_test, label="Test MSE")
        plt.xlabel(r"$Hidden Layers$", fontsize=14)
        plt.ylabel(r"$MSE$", fontsize=14)
        plt.legend(fontsize=13)
        plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
        plt.savefig(f"../../article/figures/mse_hidden_layers_node_{node}.pdf",
                    bbox_inches="tight")
        plt.show()

    num_nodes = np.arange(41, 46)
    num_layers = np.arange(2, 8)
    test_scores_NN_lay_node = np.zeros((len(num_layers), len(num_nodes)))

    #Nodes and layers
    for i, layers in enumerate(num_layers):
        for j, nodes in enumerate(num_nodes):
            print("Layers: ", layers, " ",  "Nodes: ", nodes)
            NN = NeuralNetwork(X_train, Z_train,
                               layers,
                               nodes,
                               batch_size,
                               eta,
                               lmbd,
                               gamma,
                               seed,
                               activation,
                               cost_func,
                               loss,
                               callback=False)

            NN.train_network_stochastic(int(epochs))
            test_scores_NN_lay_node[i][j] = NN.get_score(X_test, Z_test)

    plot_heatmap(test_scores_NN_lay_node, [r"Number of nodes", num_nodes],
                 [r"Number of layers", num_layers], title="Grid search Neural Network", name=None)




    num_hidden_layers = 3
    num_hidden_nodes = 42
    activation = "sigmoid"

    NN = NeuralNetwork(X_train, Z_train,
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
                       callback=False)

    NN.train_network_stochastic(int(epochs))
    print("MSE: {}".format(NN.MSE_score(X_test, Z_test)))
    print("R2: {}".format(NN.R2_score(X_test, Z_test)))


    N = 1000
    x_val = np.random.uniform(0, 1, N)
    y_val = np.random.uniform(0, 1, N)

    X_validation = np.array([x_val, y_val]).T

    # exit()
    z_val = FrankeFunction(x_val, y_val) + noise * np.random.randn(N)

    X_validation, x_val, y_val = mean_scale(X_validation, x_val, y_val)


    prediction = NN.predict(X_validation)
    print(prediction.shape)


    plot_3D_shuffled("Neural network prediction on Franke's function",
                     x_val, y_val, prediction.ravel(), "z", "test.pdf", show = True, save = True)

    plot_3D_shuffled("Input data",
                     x_val, y_val, z_val, "z", "act_data.pdf", show = True, save = True)


    "---- Tensorflow test against our neural net ----"


    epochs1 = 400
    gamma = 0.0
    cost_func="mean_squared_error"
    model = Sequential()
    model.add(Input(shape=2))
    model.add(Dense(42, input_dim=2, activation='sigmoid', kernel_regularizer=regularizers.l2(lmbd)))
    model.add(Dense(42, input_dim=2, activation='sigmoid', kernel_regularizer=regularizers.l2(lmbd)))
    model.add(Dense(42, input_dim=2, activation='sigmoid', kernel_regularizer=regularizers.l2(lmbd)))

    model.add(Dense(1, input_dim=2, activation='sigmoid'))
    sgd = optimizers.SGD(learning_rate=eta,  momentum=gamma)
    model.compile(loss=cost_func, optimizer="adam", metrics=['mean_squared_error'])

    model.fit(X_train, Z_train, epochs=epochs1, verbose=1)

    ts_pred = model.predict(X_validation)
    print("MSE: ", MSE(ts_pred.ravel(), z_val))
    print("R2: ", R2(z_val,ts_pred.ravel()))
    







if __name__ == "__main__":
    sys.path.insert(1, "../../../Project 1/code/")
    from Functions import *
    from prediction_plots import plot_3D_shuffled
    # logic_gates_NN()
    # logic_gates_OLS()
    # tensorflow_logic_gates()
    # tensorflow_copy()
    Franke_NN()
