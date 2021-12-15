from ML_PDE_solver import *
import tensorflow as tf
import ExplicitSolver as ES
import numpy as np
import time



# def time_scaling_NN(h_arr):
#     #h = dx * dt
#     # use dx = dt
#
#     L = 1
#     T = 1
#     lr = 5e-2
#     epochs = 1e3
#     time = np.zeros(len(h_arr))
#
#
#     for i in range(len(h_arr)):
#         tf.random.set_seed(123)
#
#         # define dx and dt and x,t-arrays
#         dx = np.sqrt(h[i]); dt = np.sqrt(h[i])
#         x = np.linspace(0, L, int(L / dx))
#         t = np.linspace(0, T, int(T / dt))
#         dx = x[1]-x[0]; dt = t[1]-t[0]
#         h_arr[i] = dx*dt # update h_arr to account for round off
#
#
#         with tf.device('/CPU:0'):
#             ML = NeuralNetworkPDE(x, t, int(epochs), I, lr)
#
#             # Timing
#             start = time.perf_counter()
#             loss = ML.train()
#             finish = time.perf_counter()
#             time[i] = finish - start
#
#     plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
#     plt.xlabel(r"$h$", fontsize=14)
#     plt.ylabel(r"Time", fontsize=14)
#     plt.legend(fontsize = 13)
#     plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
#     # plt.savefig("../article/figures/figure.pdf", bbox_inches="tight")
#     plt.show()
#
#
# def time_scaling_Explicit(h_arr):
#     #h = dx * dt
#
#
#     I = lambda x: np.sin(np.pi * x)
#     L = 1
#     T = 1  # 0.5
#     dx = 0.01
#     dt = 0.5 * dx ** 2
#     c = 0
#     d = 0
#
#     time = np.zeros(len(h_arr))
#
#
#     for i in range(len(h_arr)):
#         # use dt = 0.5*dx^2
#
#         dx = (2*h[i])**(1/3); dt = 1/2*(2*h[i])**(2/3)
#         x = np.linspace(0, L, int(L / dx))
#         t = np.linspace(0, T, int(T / dt))
#         dx = x[1]-x[0]; dt = t[1]-t[0]
#         h_arr[i] = dx*dt # update h_arr to account for round off
#
#         ES = ExplicitSolver(I, L, T, dx, dt, c, d)
#
#         # Timing
#         start = time.perf_counter()
#         solution = ES.run_simulation()
#         finish = time.perf_counter()
#         time[i] = finish - start
#
#     plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
#     plt.xlabel(r"$h$", fontsize=14)
#     plt.ylabel(r"Time", fontsize=14)
#     plt.legend(fontsize = 13)
#     plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
#     # plt.savefig("../article/figures/figure.pdf", bbox_inches="tight")
#     plt.show()




def create_file(filename):
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
    data = [dx, dt, epochs, t, MSE, timing, avg_runs]
    file = open(filename, 'a')
    spacing = "                  "
    for i in range(len(data)):
        file.write(f"{data[i]:5.10e}")
        file.write(spacing)
    file.write("\n")

def calculate_MSE(x, t, u):
    # Assumes solution at one given time
    u_exact = np.sin(np.pi * x) * np.exp(-np.pi**2 * t)
    MSE = np.mean((u_exact - u)**2)

    return MSE

def readfile(filename):
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

    # finite difference

    dx, dt, epochs, T, MSE, timing, avg_runs = readfile(filename)

    # fin_diff_index = np.concatenate( np.argwhere(np.isnan(epochs) == True))
    fin_diff_index = np.argwhere(np.isnan(epochs) == True)
    NN_index =  np.argwhere(np.isnan(epochs) == False)


    # print(timing[fin_diff_index])
    plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
    plt.plot(timing[fin_diff_index], MSE[fin_diff_index], 'o', label = "Finite difference")
    plt.plot(timing[NN_index], MSE[NN_index], 'o', label = "Neural network")
    plt.xlabel(r"Timing $[s]$", fontsize=14)
    plt.ylabel(r"MSE", fontsize=14)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(fontsize = 13)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    # plt.savefig("../article/figures/figure.pdf", bbox_inches="tight")
    plt.show()


def generate_data(filename, method, dx_list, avg_runs = 1, epochs = np.nan):
    # Common settings
    T = 0.2
    L = 1
    epochs

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
    # filename = "PDE_comparison.txt"
    filename = "PDE_comparison_ex.txt"
    # create_file(filename)

    #bang_for_the_buck(filename)

    # dx_list = [0.1, 0.01, 0.001, 0.0005] #finite difference
    dx_list = [0.0005] #finite difference

    # dx_list = [0.1, 0.01, 0.001] # NN
    generate_data(filename, "fin_diff", dx_list, avg_runs = 1)


    # generate_data(filename, "NN", dx_list, avg_runs = 5, epochs = 10)
    # generate_data(filename, "NN", dx_list, avg_runs = 5, epochs = 100)
    # generate_data(filename, "NN", dx_list, avg_runs = 5, epochs = 1000)
    # generate_data(filename, "NN", dx_list, avg_runs = 5, epochs = 10000)
    # generate_data(filename, "NN", dx_list, avg_runs = 5, epochs = 30000)



    exit()
