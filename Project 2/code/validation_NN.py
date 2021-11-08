


def WIP_test_logical_gates():
    # Input for logic gates
    x1 = np.array([0, 0, 1, 1])
    x2 = np.array([0, 1, 0, 1])
    X = np.array([x1, x2]).T # Design matrix

    # Types of logical gates
    y_AND = np.array([0, 0, 0, 1]) # And
    y_OR =  np.array([0, 1, 1, 1]) # Or
    y_XOR = np.array([0, 1, 1, 0]) # Exclusive OR

    # Output for logic gates
    y_gates = [y_AND, y_OR, y_XOR].reshape(4,1)


    # NN architecture
    num_hidden_layers = 1
    num_hidden_nodes = 10
    n_categories = 1

    eta = 1e-2
    lmbd = 0
    epochs = 1000
    batch_size = 4 # Full GD

    for i, y in enumerate(y_gates):
        # sklearn_pred, sklearn_accuracy = sklearn_NN(X, y_OR, eta, lmbd, epochs, num_hidden_layers, num_hidden_nodes, n_categories, 'logistic')
        NN = NeuralNetwork(X, y_gates,
                                 num_hidden_layers=2,
                                 num_hidden_nodes=10,
                                 batch_size=1,
                                 eta=0.001,
                                 lmbd=0.00,
                                 gamma = 0.0,
                                 seed=4155,
                                 activation="sigmoid",
                                 cost="MSE")


    

        NN.run_network_stochastic(epochs)
        print(np.around(NN.layers_a[-1]))
        accuracy = NN.accuracy_score()
        print(f"{int(accuracy*len(NN.t))} / {len(NN.t)} accurate predictions")

    success = True
    msg = "..."




if __name__ == "__main__":
