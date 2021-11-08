
def standard_scale(*args):
    scaled = []
    scaler = StandardScaler()
    for arg in args:
        scaler.fit(arg)
        scaled.append(scaler.transform(arg))

    if len(args) == 1:  # If just one argument
        return scaled[0]
    else:
        return scaled

def mean_scale_new(*args):
    scaled = []
    for arg in args:
        arg =  arg - np.mean(arg, axis=0)
        scaled.append(arg)

    if len(args) == 1:  # If just one argument
        return scaled[0]
    else:
        return scaled



def find_hyperparameters(X,
                         z,
                         epochs,
                         num_hidden_layers,
                         num_hidden_nodes,
                         batch_size,
                         etas,
                         lmbds,
                         activation,
                         cost_func,
                         name = 0,
                         scaling = "std",
                         return_best = False
                         ):
    from sklearn.model_selection import train_test_split
    from find_hyperparameters import find_hyperparameters
    from sklearn.preprocessing import StandardScaler
    from NeuralNetwork import NeuralNetwork
    import seaborn as sns
    import numpy as np
    import matplotlib.ticker as tkr
    import matplotlib.pyplot as plt



    sns.set()
    #Split and scale data if choosen
    X_train, X_test, Z_train, Z_test = train_test_split(X, z, test_size=0.2)

    if scaling == "std":
        X_train, X_test, Z_train, Z_test = standard_scale(X_train, X_test, Z_train, Z_test)
    elif scaling == "mean":
        X_train, X_test, Z_train, Z_test = mean_scale_new(X_train, X_test, Z_train, Z_test)



    #Setup arrays for accuracy score
    test_accuracy = np.zeros((len(etas), len(lmbds)))

    for i, eta in enumerate(etas):
        for j, lmbd in enumerate(lmbds):
            NN = NeuralNetwork( X_train,
                                Z_train,
                                num_hidden_layers,
                                num_hidden_nodes,
                                batch_size,
                                eta,
                                lmbd,
                                activation = activation,
                                cost = cost_func)

            NN.train_network_stochastic(int(epochs))
            test_accuracy[i][j] = NN.accuracy_score(X_test, Z_test)

    fig, ax = plt.subplots(figsize = (10, 10))
    print(test_accuracy)
    sns.heatmap(test_accuracy, xticklabels = np.log10(lmbds), yticklabels = np.log10(etas), annot=True, ax=ax, cmap="viridis")
    #ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda y, p: f'{y:.1f}'))
    #ax.xaxis.set_major_formatter(tkr.FuncFormatter(lambda x, p: f'{x:.1f}'))
    ax.set_title("Test Accuracy")
    ax.set_ylabel("$log_{10}(\eta)$")
    ax.set_xlabel("$log_{10}(\lambda)$")
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    plt.subplots_adjust(hspace=0.3)

    if isinstance(name, str):
        plt.savefig(f"../article/figures/hyper_param_{name}_{activation}.pdf",
                bbox_inches="tight")


    indx = np.where(test_accuracy == np.max(test_accuracy))
    print(etas[int(indx[0][0])], lmbds[int(indx[1][0])])
    plt.show()

    if return_best:
        indx = np.where(test_accuracy == np.max(test_accuracy))
        return etas[int(indx[0][0])], lmbds[int(indx[1][0])], np.max(test_accuracy)
