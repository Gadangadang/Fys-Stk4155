import numpy as np
import matplotlib.pyplot as plt

from NeuralNetwork import NeuralNetwork
# from find_hyperparameters import *
import os
import sys
path = os.getcwd()  # Current working directory
path += '/../../Project 1/code'
sys.path.append(path)
from Functions import *

# sklearn classifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Tensorflow
from tensorflow import keras
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential      #This allows appending layers to existing models
from tensorflow.keras.layers import Dense           #This allows defining the characteristics of a particular layer
from tensorflow.keras import optimizers             #This allows using whichever optimiser we want (sgd,adam,RMSprop)
from tensorflow.keras import regularizers           #This allows using whichever regularizer we want (l1,l2,l1_l2)


def logic_gates():
    # Input
    x1 = np.array([0, 0, 1, 1])
    x2 = np.array([0, 1, 0, 1])
    X = np.array([x1, x2]).T # Design matrix

    # Output
    y_AND = np.array([0, 0, 0, 1]) # And
    y_OR =  np.array([0, 1, 1, 1]) # Or
    y_XOR = np.array([0, 1, 1, 0]) # Exclusive OR
    y_gates = np.array([y_AND, y_OR, y_XOR]).T


    # NN architecture
    num_hidden_layers = 1
    num_hidden_nodes = 16
    batch_size = 4 # Full GD
    n_categories = 1
    eta = 1e-1
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
                             loss = "accuracy",
                             callback = True)

    NN.train_network_stochastic(epochs, plot = True)


    exit()
    # sklearn_pred, sklearn_accuracy = sklearn_NN(X, y.ravel(), eta, lmbd, epochs, num_hidden_layers, num_hidden_nodes, n_categories, 'logistic')
    NN = NeuralNetwork(X, y,
                             num_hidden_layers,
                             num_hidden_nodes,
                             batch_size,
                             eta,
                             lmbd,
                             gamma,
                             seed=4155,
                             activation=activation,
                             cost=cost_func)


    NN.train_network_stochastic(epochs)

    exit()
    accuracy = NN.accuracy_score(X, y)
    NN_pred = np.round(NN.predict(X)).ravel()
    NN_pred = NN.predict(X)


    print("sklearn, acc:", sklearn_accuracy)
    print("sklearn, pred:", sklearn_pred)

    print("NN, acc:", accuracy)
    print("NN, pred:", NN_pred)


def Franke_NN():
    print("work here sakki")

    np.random.seed(0)
    noise = 0.2
    N = 100
    x = np.random.uniform(0, 1, N)
    y = np.random.uniform(0, 1, N)

    X = np.array([x,y]).T

    #exit()
    z = FrankeFunction(x, y) + noise * np.random.randn(N)
    z = z.reshape(N, 1)

    X_train, X_test, Z_train, Z_test = train_test_split(X, z, test_size=0.2)
    X_train, X_test = mean_scale(X_train, X_test)

    print(X_train.shape, Z_train.shape)

    eta = 0.001
    epochs = 100
    gamma = 0
    lmbd = 0.0
    lay = 2
    nodes = 50
    batch_size = 15

    NN = NeuralNetwork(X_train,
                       Z_train,
                       num_hidden_layers=lay,
                       num_hidden_nodes=nodes,
                       batch_size=batch_size,
                       eta=eta,
                       lmbd=lmbd,
                       gamma = gamma,
                       seed=4155,
                       activation="sigmoid",
                       cost="MSE",
                       loss = "R2",
                       callback = False)


    NN.train_network_stochastic(epochs, plot = False)
    print("R2 score : ", NN.get_score(X_test, Z_test))

    NN = NeuralNetwork(X_train,
                       Z_train,
                       num_hidden_layers=lay,
                       num_hidden_nodes=nodes,
                       batch_size=batch_size,
                       eta=eta,
                       lmbd=lmbd,
                       gamma = gamma,
                       seed=4155,
                       activation="sigmoid",
                       cost="MSE",
                       loss = "MSE",
                       callback = False)


    NN.train_network_stochastic(epochs, plot = False)
    print("MSE score : ", NN.get_score(X_test, Z_test))










def sklearn_NN(X, y, eta, lmbd, epochs, num_hidden_layers, num_hidden_nodes, n_categories, activation_func):
    n_inputs, n_features = X.shape
    hidden_layer_sizes = [num_hidden_nodes for i in range(num_hidden_layers)]
    dnn = MLPClassifier(hidden_layer_sizes =  hidden_layer_sizes,
                        activation = activation_func,
                        alpha= lmbd,
                        learning_rate_init = eta,
                        max_iter = epochs )
    dnn.fit(X, y)
    test_pred = dnn.predict(X)
    test_accuracy = accuracy_score(y, test_pred)
    return test_pred, test_accuracy




def tensorflow_logic_gates():
    # Input for logic gates
    x1 = np.array([0, 0, 1, 1])
    x2 = np.array([0, 1, 0, 1])
    X = np.array([x1, x2], dtype=np.float64).T # Design matrix


    # Types of logical gates
    y_AND = np.array([0, 0, 0, 1]) # And
    y_OR =  np.array([0, 1, 1, 1]) # Or
    y_XOR = np.array([0, 1, 1, 0]) # Exclusive OR

    # Output for logic gates
    y_gates = [y_AND, y_OR, y_XOR]


    #
    # X = np.array([ [0, 0], [0, 1], [1, 0],[1, 1]],dtype=np.float64)
    # y_XOR = np.array( [ 0, 1 ,1, 0])
    # y_OR = np.array( [ 0, 1 ,1, 1])
    # y_AND = np.array( [ 0, 0 ,0, 1])

    # Architecture
    y = y_OR
    num_hidden_nodes = 16
    batch_size = 4 # Full GD
    n_categories = 1
    eta = 1e-1
    lmbd = 0
    gamma = 0.9

    activation = "relu"
    # cost_func = "categorical_crossentropy"
    cost_func = "mean_squared_error"
    # cost_func = "binary_crossentropy"
    epochs = 500

    # Tensorflow (tf) network
    model = Sequential()
    model.add(Input(shape=2))
    model.add(Dense(num_hidden_nodes, activation=activation, kernel_regularizer=regularizers.l2(lmbd)))
    # model.add(Dense(num_hidden_nodes, activation=activation, kernel_regularizer=regularizers.l2(lmbd)))
    model.add(Dense(n_categories))

    sgd = optimizers.SGD(learning_rate=eta,  momentum=gamma)
    model.compile(loss=cost_func, optimizer=sgd, metrics=['accuracy'])
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=2)
    # model.summary()

    # Evaluate results
    tf_pred = model.predict(X) # Tensorflow prediction

    print(tf_pred)
    # print(np.round(tf_pred))
    # score = accuracy_score(y, np.round(tf_pred))
    # print(score)




def tensorflow_copy():
    import numpy as np
    from keras.models import Sequential
    from keras.layers.core import Dense

    # the four different states of the XOR gate
    training_data = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")

    # the four expected results in the same order
    target_data = np.array([[0],[1],[1],[0]], "float32")

    model = Sequential()
    model.add(Dense(16, input_dim=2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['binary_accuracy'])

    model.fit(training_data, target_data, epochs=100, verbose=2)

    print(model.predict(training_data).round())



def multiple_categories_test():
    # Input for logic gates
    x1 = np.array([0, 0, 1, 1])
    x2 = np.array([0, 1, 0, 1])
    X = np.array([x1, x2]).T # Design matrix

    y = np.array([[0, 0, 0, 1], [0, 1, 1, 1], [0, 1, 1, 0]]).T #AND, OR, XOR


    # NN architecture
    num_hidden_layers = 1
    num_hidden_nodes = 2
    batch_size = 4 # Full GD
    eta = 1
    lmbd = 1e-5
    gamma = 0

    activation = "sigmoid"
    # cost_func = "MSE"
    cost_func = "cross_entropy"
    epochs = 1000

    y = y[:,2].reshape(4,1)

    etas = np.logspace(1,-1,3)
    lmbds = np.logspace(-2,-8,7)

    # etas = np.logspace(2,-5,8)
    # lmbds = np.logspace(-2,-8,7)

    # eta, lmbd, acc = find_hyperparameters(X, X, y, y,
    #                                         num_hidden_layers,
    #                                         num_hidden_nodes,
    #                                         batch_size,
    #                                         etas,
    #                                         lmbds,
    #                                         gamma,
    #                                         epochs,
    #                                         activation=activation,
    #                                         cost=cost_func,
    #                                         seed=4155,
    #                                         name = 0,
    #                                         return_best = True)
    #
    #
    # exit()
    NN = NeuralNetwork(X, y,
                             num_hidden_layers,
                             num_hidden_nodes,
                             batch_size,
                             eta,
                             lmbd,
                             gamma,
                             seed=4155,
                             activation=activation,
                             cost=cost_func,
                             callback = "accuracy")

    NN.train_network_stochastic(epochs, plot = False)
    accuracy = NN.accuracy_score(NN.X, NN.T)
    NN_pred = np.round(NN.predict(X)).ravel()
    NN_pred = NN.predict(NN.X)

    print(accuracy)
    print(NN_pred)


def XOR_manuel():


    # NN architecture
    num_hidden_layers = 1
    num_hidden_nodes = 2
    batch_size = 5 # Full GD
    eta = 1
    lmbd = 1e-5
    gamma = 0

    activation = "sigmoid"
    # cost_func = "MSE"
    cost_func = "cross_entropy"
    epochs = 1000



    x1 = np.array([0, 0, 1, 1])
    x2 = np.array([0, 1, 0, 1])
    X = np.array([x1, x2]).T # Design matrix

    y_XOR = np.array([0, 1, 1, 0]).reshape(4,1)


    NN = NeuralNetwork(X, y_XOR, num_hidden_layers = 1, num_hidden_nodes = 2)
    #
    # print(NN.weights)
    # print(NN.bias)











if __name__ == "__main__":
    #logic_gates()
    # tensorflow_logic_gates()
    # morten_test()
    # tensorflow_copy()
    # multiple_categories_test()
    # XOR_manuel()
    Franke_NN()
