import numpy as np
import matplotlib.pyplot as plt

from NeuralNetwork import NeuralNetwork
from find_hyperparameters import *

# sklearn classifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Tensorflow
from tensorflow import keras
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential      #This allows appending layers to existing models
from tensorflow.keras.layers import Dense           #This allows defining the characteristics of a particular layer
from tensorflow.keras import optimizers             #This allows using whichever optimiser we want (sgd,adam,RMSprop)
from tensorflow.keras import regularizers           #This allows using whichever regularizer we want (l1,l2,l1_l2)



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

def logic_gates():
    # Input for logic gates
    x1 = np.array([0, 0, 1, 1])
    x2 = np.array([0, 1, 0, 1])
    X = np.array([x1, x2]).T # Design matrix


    # Types of logical gates
    y_AND = np.array([0, 0, 0, 1]) # And
    y_OR =  np.array([0, 1, 1, 1]) # Or
    y_XOR = np.array([0, 1, 1, 0]) # Exclusive OR

    # Output for logic gates
    y_gates = [y_AND, y_OR, y_XOR]
    y_gates = [y_XOR]




    # NN architecture
    num_hidden_layers = 1
    num_hidden_nodes = 16
    batch_size = 4 # Full GD
    n_categories = 1
    eta = 1e-1
    lmbd = 0
    gamma = 0


    etas = np.logspace(-1, 0, 3)
    # etas = [2]
    lmbds = np.logspace(-8, -6, 3)

    activation = "sigmoid"
    # cost_func = "cross_entropy"
    cost_func = "MSE"
    epochs = 2000

    y = y_XOR.reshape(4,1)


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

    NN.train_network_stochastic(epochs, plot = True)


    exit()
    find_hyperparameters(X, X, y, y,
                             epochs,
                             num_hidden_layers,
                             num_hidden_nodes,
                             batch_size,
                             etas,
                             lmbds,
                             activation,
                             cost_func,
                             name = 0,
                             return_best = False)


    exit()
    for i, y in enumerate(y_gates):
        # sklearn_pred, sklearn_accuracy = sklearn_NN(X, y_OR, eta, lmbd, epochs, num_hidden_layers, num_hidden_nodes, n_categories, 'logistic')
        y = y.reshape(4,1)
        input()


        # find_hyperparameters(X,
        #                          y,
        #                          epochs,
        #                          num_hidden_layers,
        #                          num_hidden_nodes,
        #                          batch_size,
        #                          etas,
        #                          lmbds,
        #                          activation,
        #                          cost_func,
        #                          name = 0,
        #                          scaling = "none",
        #                          return_best = False)


        sklearn_pred, sklearn_accuracy = sklearn_NN(X, y.ravel(), eta, lmbd, epochs, num_hidden_layers, num_hidden_nodes, n_categories, 'logistic')
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
        accuracy = NN.accuracy_score(X, y)
        NN_pred = np.round(NN.predict(X)).ravel()
        NN_pred = NN.predict(X)


        print("sklearn, acc:", sklearn_accuracy)
        print("sklearn, pred:", sklearn_pred)

        print("NN, acc:", accuracy)
        print("NN, pred:", NN_pred)



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


# def morten_test():
#
#     # Design matrix
#     X = np.array([ [1, 0, 0], [1, 0, 1], [1, 1, 0],[1, 1, 1]],dtype=np.float64)
#     yXOR = np.array( [ 0, 1 ,1, 0])
#     yOR = np.array( [ 0, 1 ,1, 1])
#     yAND = np.array( [ 0, 0 ,0, 1])
#
#
#
#     from sklearn.neural_network import MLPClassifier
#     from sklearn.datasets import make_classification
#     X, yXOR = make_classification(n_samples=100, random_state=1)
#
#     FFNN = MLPClassifier(random_state=1, max_iter=300).fit(X, yXOR)
#     FFNN.predict_proba(X)
#     print(f"Test set accuracy with Feed Forward Neural Network  for XOR gate:{FFNN.score(X, yXOR)}")
#
#



def multiple_categories_test():
    # Input for logic gates
    x1 = np.array([0, 0, 1, 1])
    x2 = np.array([0, 1, 0, 1])
    X = np.array([x1, x2]).T # Design matrix


    # Types of logical gates
    y_AND = np.array([0, 0, 0, 1]) # And
    y_OR =  np.array([0, 1, 1, 1]) # Or

    y = np.array([[0, 0, 0, 1], [0, 1, 1, 1]]).T


    # NN architecture
    num_hidden_layers = 1
    num_hidden_nodes = 40
    batch_size = 4 # Full GD
    eta = 1
    lmbd = 0
    gamma = 0

    activation = "sigmoid"
    cost_func = "MSE"
    # cost_func
    epochs = 1000

    # y = y[:,0].reshape(4,1)
    # y = y[:,1].reshape(4,1)
    # print(y)
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
                             callback = "accuracy__")

    NN.train_network_stochastic(epochs, plot = False)
    accuracy = NN.accuracy_score(NN.X, NN.T)
    NN_pred = np.round(NN.predict(X)).ravel()
    NN_pred = NN.predict(NN.X)
    print(accuracy)
    print(NN_pred)





if __name__ == "__main__":
    # logic_gates()
    # tensorflow_logic_gates()
    # morten_test()
    # tensorflow_copy()

    multiple_categories_test()
