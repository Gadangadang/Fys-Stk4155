import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from SGD import *
from Breast_cancer import find_hyperparameters

def plot_digits(digits):
    fig,axes = plt.subplots(nrows= 1, ncols=4, figsize=(10, 3))
    for ax, image, label in zip(axes, digits.images, digits.target):
        ax.axis("off")
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Training: %i" % label)
    plt.show()

if __name__ == "__main__":
    digits = datasets.load_digits()
    #plot_digits(digits)

    # flatten the images
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

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
    n_categories = 1
    eta = 1e-2
    lmbd = 0.5
    epochs = int(50)
    batch_size = int(100)

    etas = np.logspace(-4, 1, 5)
    lmbds = np.logspace(-4, 1, 5)

    activation = "sigmoid"
    cost_func = "cross_entropy"
    name="breast_cancer"
    best_eta, best_lmbd, best_val = find_hyperparameters(X,
                                                         y,
                                                         epochs,
                                                         num_hidden_layers,
                                                         num_hidden_nodes,
                                                         batch_size,
                                                         etas,
                                                         lmbds,
                                                         activation,
                                                         cost_func,
                                                         name,
                                                         return_best = True)
    print(f"{100*best_val:.0f}% NN accuracy. ")
