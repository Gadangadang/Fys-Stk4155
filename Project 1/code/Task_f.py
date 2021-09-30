import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from prediction_plots import plot_3D


# Load the terrain
terrain1 = imread("../article/Dagestan.tif")
# Show the terrain
title = "Terrain over Dagestan 1"
length = np.shape(terrain1)[0]
x = np.linspace(0, length-1, length)
y = np.linspace(0, length-1, length)
X,Y = np.meshgrid(x,y)
print(np.shape(X))

z = terrain1
print(np.shape(z))
z_label = "Altitude"
save_name = "Dagestan_terrain"
plot_3D(title, X, Y, z, z_label, save_name)
