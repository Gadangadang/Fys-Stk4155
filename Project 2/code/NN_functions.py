from sklearn.preprocessing import StandardScaler
import seaborn as sns
import numpy as np
import matplotlib.ticker as tkr
import matplotlib.pyplot as plt

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

def plot_heatmap(test_scores, param1, param2, title = None, name = None):
    # Heatmap
    param1_label, param1_vals = param1
    param2_label, param2_vals = param2
    fig, ax = plt.subplots(figsize = (7, 7))
    sns.set()
    sns.heatmap(test_scores, xticklabels = param1_vals, yticklabels = param2_vals, annot=True, ax=ax)
    if isinstance(title, str):
        ax.set_title(title)
    else:
        ax.set_title("Test accuracy")
    ax.set_xlabel(param1_label)
    ax.set_ylabel(param2_label)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    plt.subplots_adjust(hspace=0.3)
    if isinstance(name, str):
        plt.savefig(f"../article/figures/hyper_param_{name}_{activation}.pdf",
                bbox_inches="tight")

    plt.show()

def plot_digits(digits):
    fig,axes = plt.subplots(nrows= 1, ncols=4, figsize=(10, 3))
    for ax, image, label in zip(axes, digits.images, digits.target):
        ax.axis("off")
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Training: %i" % label)
    plt.show()



#Setup arrays for accuracy score
#test_accuracy = np.zeros((len(etas), len(lmbds)))

#for i, param1_i in enumerate(param1):
#    for j, param1_j in enumerate(param2):
#        print(f"\r(eta_val, lmbd_val) = ({i},{j})/({len(etas)-1},{len(lmbds)-1})", end="")
#        NN = NeuralNetwork(X_train,
#                           y_train,
#                           num_hidden_layers,
#                           num_hidden_nodes,
#                           batch_size=batch_size,
#                           eta,
#                           lmbd,
#                           gamma,
#                           seed,
#                           activation,
#                           cost,
#                           loss,
#                           callback = True)
#
#        NN.train_network_stochastic(int(epochs))
#        test_accuracy[i][j] = NN.accuracy_score(X_test, y_test)
#indx = np.where(test_accuracy == np.max(test_accuracy))