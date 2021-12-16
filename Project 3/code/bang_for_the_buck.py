from ML_PDE_solver import *
import tensorflow as tf
import ExplicitSolver as ES
import numpy as np
import time
import pandas as pd





def create_file(filename):
    """
    Create file with format for collecting data.

    Args:
        filename (string): filename
    """
    header = ["dx", "dt", "epochs", "T", "MSE", "timing[s]", "avg_runs"]
    create = input(f"Are you sure you want to create {filename}?, y/n: ")
    if create == "y":
        file = open(filename, 'w')
        spacing = "                  "
        for i in range(len(header)):
            file.write(header[i])
            file.write(spacing)
        file.write("\n")
        print(f"Created: {filename}")

    else:
        print("Aborting")
        exit()

def append_to_file(dx, dt, epochs, t, MSE, timing, avg_runs, filename):
    """
    Append data to file.

    Args:
        dx (float): spacial step
        dt (float): time step
        epochs (int): number of epochs
        t (float): time point
        MSE (float): mean squared error
        timing (float): computation time
        avg_runs (int): number of runs to average over
        filename (string): filename to append the data to
    """
    data = [dx, dt, epochs, t, MSE, timing, avg_runs]
    file = open(filename, 'a')
    spacing = "                  "
    for i in range(len(data)):
        file.write(f"{data[i]:5.10e}")
        file.write(spacing)
    file.write("\n")

def calculate_MSE(x, t, u):
    """
    Calculate the MSE.

    Args:
        x (array): spacial domain
        t (array): time domain
        u (array): solution array

    Returns:
        [float]: mean square error
    """
    # Assumes solution at one given time
    u_exact = np.sin(np.pi * x) * np.exp(-np.pi**2 * t)
    MSE = np.mean((u_exact - u)**2)

    return MSE

def readfile(filename):
    """
    Fetch data from given file.

    Args:
        filename (string): name of the file to read

    Returns:
        [array]: data collection read from file
    """
    infile = open(filename, 'r')
    line = infile.readline()
    data = [[] for i in range(7)] #dx, dt, epochs, T, MSE, timing, avg_runs
    for line in infile:
        words = line.split()
        for i, list in enumerate(data):
            list.append(float(words[i]))
    for i, list in enumerate(data):
        data[i] = np.array(list)


    return data




def bang_for_the_buck(filename):
    """
    Plots the MSE error for the stored solutions vs computation time.

    Args:
        filename (string): name of file to read from
    """
    dx, dt, epochs, T, MSE, timing, avg_runs = readfile(filename)
    fin_diff_index = np.argwhere(np.isnan(epochs) == True)
    NN_index =  np.argwhere(np.isnan(epochs) == False)

    plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')



    size = -200/np.log10(dx)

    # Linear fit
    a_fin, b_fin, a_err_fin, b_err_fin = lin_fit(np.log10(timing[fin_diff_index]), np.log10(MSE[fin_diff_index]))
    x_fin = [np.min(timing[fin_diff_index]), np.max(timing[fin_diff_index])]
    plt.plot(x_fin, x_fin**a_fin*10**b_fin, linestyle = "--", color = color_cycle(0))

    a_NN, b_NN, a_err_NN, b_err_NN = lin_fit(np.log10(timing[NN_index]), np.log10(MSE[NN_index]))
    x_NN = [np.min(timing[NN_index]), np.max(timing[NN_index])]
    plt.plot(x_NN, x_NN**a_NN*10**b_NN, linestyle = "--", color = color_cycle(1))

    # Finite Difference
    fin_color = [color_cycle(0)]*len(fin_diff_index)
    df_fin = pd.DataFrame({
                        'X': timing[fin_diff_index].ravel(),
                        'Y': MSE[fin_diff_index].ravel(),
                        'colors': fin_color,
                        "bubble_size": size[fin_diff_index].ravel()})

    fin_scatter = plt.scatter('X', 'Y',
             s='bubble_size',
             c ='colors',
             alpha=0.8 , data=df_fin, label = "Finite difference")


    # Neural Network
    NN_color = epochs[NN_index].ravel()

    df_NN = pd.DataFrame({
                        'X': timing[NN_index].ravel(),
                        'Y': MSE[NN_index].ravel(),
                        'colors': NN_color,
                        "bubble_size": size[NN_index].ravel()})

    cmap = plt.get_cmap('Reds', 5)
    NN_scatter = plt.scatter('X', 'Y',
             s='bubble_size',
             c ='colors',
             cmap = cmap,
             alpha=0.8 , edgecolors='black', norm=matplotlib.colors.LogNorm(), data=df_NN, label = "Neural network")

    # Plot settings
    cbar = plt.colorbar()
    cbar.set_label('Epochs', rotation=270, labelpad = 20, fontsize = 16)

    plt.xlabel(r"Avg. computation time $[s]$", fontsize=16)
    plt.ylabel(r"MSE", fontsize=16)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(fontsize = 14)

    ax = plt.gca()
    leg = ax.get_legend()
    leg.legendHandles[1].set_color(color_cycle(1))

    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    plt.savefig(
        f"../article/figures/bang_for_the_buck.pdf", bbox_inches="tight")
    plt.show()


def generate_data(filename, method, dx_list, avg_runs = 1, epochs = np.nan):
    """
    Generate data and append to file.

    Args:
        filename (string): file to append data to
        method (string): specify whether to use finite difference of neural network
        dx_list (list): list of spacial steps dx
        avg_runs (int): number of runs to average over
        epochs (int): number of epochs to run for (nan for finite difference)
    """
    # Common settings
    T = 0.05
    L = 1

    MSE = np.zeros(avg_runs)
    timing = np.zeros(avg_runs)

    if method == "fin_diff":
        epochs = np.nan
        for dx in dx_list:
            for i in range(avg_runs):
                print(f"\r method: {method}, run: {i+1}/{avg_runs}, dx = {dx}, epochs = {epochs}", end="")
                I = lambda x: np.sin(np.pi * x)
                dt = 0.5 * dx**2
                c = 0
                d = 0
                ESS = ES.ExplicitSolver(I, L, T, dx, dt, c, d)
                x = ESS.x
                t = ESS.t

                #Timing
                start = time.perf_counter()
                u = ESS.run_simulation()
                finish = time.perf_counter()
                timing[i] = finish - start

                # Error
                MSE[i] = calculate_MSE(x, T, u[-1])

            print() # linebreak
            append_to_file(dx, dt, epochs, T, np.mean(MSE), np.mean(timing), avg_runs, filename)



    if method == "NN":
        epochs = epochs
        if np.isnan(epochs):
            print("Please specify epochs (got nan)")
            exit(0)
        for dx in dx_list:
            for i in range(avg_runs):
                print(f"\r method: {method}, run: {i+1}/{avg_runs}, dx = {dx}, epochs = {epochs}", end="")
                I = lambda x: tf.sin(np.pi * x)
                dt = dx*T

                lr = 5e-2
                x = np.linspace(0, L, round(L / dx) + 1)
                t = np.linspace(0, T, round(T / dt) + 1)

                # Place tensors on the CPU
                with tf.device("/CPU:0"):  # Write '/GPU:0' for large networks
                    ML = NeuralNetworkPDE(x, t, int(epochs), I, lr)

                    #Timing
                    start = time.perf_counter()
                    loss = ML.train()
                    finish = time.perf_counter()
                    timing[i] = finish - start
                    u = ML()

                # Error
                MSE[i] = calculate_MSE(x, T, u[-1])

            print() # linebreak
            append_to_file(dx, dt, epochs, T, np.mean(MSE), np.mean(timing), avg_runs, filename)

    print("DONE")





if __name__ == "__main__":
    filename = "PDE_comparison_t0.5.txt"
    # create_file(filename)

    bang_for_the_buck(filename)



    dx_list = [0.1, 0.05, 0.01, 0.005, 0.001]
    generate_data(filename, "fin_diff", dx_list, avg_runs = 5, epochs = 10)
    generate_data(filename, "NN", dx_list, avg_runs = 5, epochs = 10)

    dx_list = [0.1, 0.05, 0.01, 0.005]
    generate_data(filename, "NN", dx_list, avg_runs = 5, epochs = 100)

    dx_list = [0.1, 0.05, 0.01]
    generate_data(filename, "NN", dx_list, avg_runs = 5, epochs = 1000)

    dx_list = [0.1, 0.05]
    generate_data(filename, "NN", dx_list, avg_runs = 5, epochs = 1000)
    generate_data(filename, "NN", dx_list, avg_runs = 5, epochs = 10000)

    dx_list = [0.1]
    generate_data(filename, "NN", dx_list, avg_runs = 5, epochs = 100000)
