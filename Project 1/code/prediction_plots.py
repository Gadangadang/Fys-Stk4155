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

def plot_3D(title, x, y, z, z_label, save_name):
    fig = plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
    ax = fig.gca(projection='3d')
    plt.title(title)

    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Customize the z axis.
    #ax.set_zlim(-0.10, 1.40)
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
    plt.savefig(f"../article/figures/zprediction/{save_name}.pdf", bbox_inches="tight")
    plt.show()
    plt.clf()
