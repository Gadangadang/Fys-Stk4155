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

def plot_3D(title, x, y, z, z_label, save_name, show = False, save = True):
    """Creates 3D plot

    Args:
        title (String): Title of plot
        x          (Array): x mesh grid
        y          (Array): y meshgrid
        z          (Array): Mesh grid with the data
        z_label   (String): Label for the z axis component
        save_name (String): Name to save file as
    """
    fig = plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
    ax = fig.gca(projection='3d')
    plt.title(title)

    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                           linewidth=0.2, antialiased=False)

    # Customize the z axis.
    if not show:
        ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.zaxis.set_rotate_label(False)
    ax.set_xlabel(r"$x$", fontsize=14)
    ax.set_ylabel(r"$y$", fontsize=14)
    ax.set_zlabel(z_label, fontsize=14)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout(pad=1.5, w_pad=0.7, h_pad=0.2)
    ax.view_init(azim=45)
    if save:
        plt.savefig(f"../article/figures/zprediction/{save_name}.pdf", bbox_inches="tight")
        plt.clf()
    if show:
        plt.show()


def plot_3D_shuffled(title, x, y, z, z_label, save_name, show = False, save = True):
    """Creates 3D plot

    Args:
        title (String): Title of plot
        x          (Array): x flat
        y          (Array): y flat
        z          (Array): z flat
        z_label   (String): Label for the z axis component
        save_name (String): Name to save file as
    """
    fig = plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
    ax = fig.gca(projection='3d')
    plt.title(title)

    # Plot the surface.
    # surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
    #                        linewidth=0.2, antialiased=False)
    # surf = ax.plot_trisurf(x, y, z,
    #             cmap=cm.coolwarm, edgecolor='none');
    surf = ax.plot_trisurf(x, y, z,
                cmap=cm.coolwarm, edgecolor='darkgrey',  linewidth=0.2, antialiased=True)



    # Customize the z axis.

    # ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(4))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.zaxis.set_rotate_label(False)
    ax.set_xlabel(r"$x$", fontsize=14)
    ax.set_ylabel(r"$y$", fontsize=14)
    ax.set_zlabel(z_label, labelpad= 10, fontsize=14)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.tight_layout(pad=1.5, w_pad=0.7, h_pad=0.2)
    ax.view_init(azim=70)
    if save:
        plt.savefig(f"../article/figures/zprediction_real_data/{save_name}.pdf", bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.clf()
