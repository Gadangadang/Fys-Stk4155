from import_folders import *
import_all_folders()
from NeuralNetwork import NeuralNetwork
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
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
        arg = arg - np.mean(arg, axis=0)
        scaled.append(arg)

    if len(args) == 1:  # If just one argument
        return scaled[0]
    else:
        return scaled


def plot_heatmap(test_scores, param1, param2, title=None, name=None):
    # Heatmap
    param1_label, param1_vals = param1
    param2_label, param2_vals = param2
    fig, ax = plt.subplots(figsize=(7, 7))
    sns.set()
    sns.heatmap(test_scores, xticklabels=np.around(param1_vals, 2),
                yticklabels=np.around(param2_vals, 2), annot=True, ax=ax)
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
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, label in zip(axes, digits.images, digits.target):
        ax.axis("off")
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(label, fontsize=20)
    plt.show()


def plot_predictions(X_test, predict, Y_test):
    fig = plt.figure(figsize=(6, 6))
    fig.subplots_adjust(left=0, right=1, bottom=0,
                        top=1, hspace=0.05, wspace=0.05)
    for i in range(64):
        ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
        ax.imshow(X_test.reshape(-1, 8, 8)[i], cmap=plt.cm.binary,
                  interpolation='nearest')
        # label the image with the target value
        if predict[i] == Y_test[i]:
            ax.text(0, 7, str(predict[i]), color='green', fontsize=16)
        else:
            ax.text(0, 7, str(predict[i]), color='red', fontsize=16)

    plt.show()
def sklearn_NN(X, y, eta, lmbd, epochs, num_hidden_layers, num_hidden_nodes, n_categories, activation_func):
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    hidden_layer_sizes = [num_hidden_nodes for i in range(num_hidden_layers)]

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    scaler.fit(X_test)
    X_test = scaler.transform(X_test)

    dnn = MLPClassifier(hidden_layer_sizes =  hidden_layer_sizes,
                        activation = activation_func,
                        alpha= lmbd,
                        learning_rate_init = eta,
                        max_iter = epochs   )
    dnn.fit(X_train, Y_train)
    test_pred = dnn.predict(X_test)
    test_accuracy = accuracy_score(Y_test, test_pred)
    return test_pred, test_accuracy



if __name__ == "__main__":
    pass
