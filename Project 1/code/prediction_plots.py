import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from Functions import *

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from Functions import *

from tqdm import trange


# from plot_set import * # Specifies plotting settings


def show_predictions(N, z_noise, n, seed = 4155):
    """
    Show 3D plot of z-model/predictions
    Makes one plot for each n
    This one is actually best without plot_set
    """


    x, y, z = generate_data(N, z_noise, seed)


    #--- Plot input data ---#
    fig = plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
    ax = fig.gca(projection='3d')
    plt.title(f"Input data")

    # Plot the surface.
    surf = ax.plot_surface(x, y, z.reshape(N, N), cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.zaxis.set_rotate_label(False)
    ax.set_xlabel(r"$x$", fontsize=14)
    ax.set_ylabel(r"$y$", fontsize=14)
    ax.set_zlabel(r"$z$", fontsize=14)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout(pad=1.5, w_pad=0.7, h_pad=0.2)
    ax.view_init(azim=45)
    plt.savefig(f"../article/figures/ztilde/input_data.pdf", bbox_inches="tight")
    plt.clf()


    z_mean = np.mean(z)
    mean_scale(z)
    for i in trange(len(n)):
        X = create_X(x, y, n[i])
        mean_scale(X)

        beta_OLS = OLS_regression(X, z)
        ztilde = (X @ beta_OLS).ravel()
        ztilde = ztilde.reshape(N, N) + z_mean

        fig = plt.figure(num=i, dpi=80, facecolor='w', edgecolor='k')
        ax = fig.gca(projection='3d')
        plt.title(f"n = {n[i]}")

        # Plot the surface.
        surf = ax.plot_surface(x, y, ztilde, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        # Customize the z axis.
        ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        ax.zaxis.set_rotate_label(False)
        ax.set_xlabel(r"$x$", fontsize=14)
        ax.set_ylabel(r"$y$", fontsize=14)
        ax.set_zlabel(r"$\tilde{z}$", fontsize=14)

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.tight_layout(pad=1.5, w_pad=0.7, h_pad=0.2)
        ax.view_init(azim=45)
        plt.savefig(f"../article/figures/ztilde/ztilde_n{n[i]}.pdf", bbox_inches="tight")
        plt.clf()
