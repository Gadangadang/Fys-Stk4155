import matplotlib.pyplot as plt

def plot_difference(data1, name1, data2, name2, x, t):

    error_1 = rel_err_total(data1, x, t)
    error_2 = rel_err_total(data2, x, t)


    plt.plot(t, error_1, label=f"{name1} error")
    plt.plot(t, error_2, label=f"{name2} error")
    plt.xlabel("Time t")
    plt.ylabel("Error")
    plt.title(f"Mean Error vs Exact as function of time")
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    plt.legend()
    plt.show()

def loss_plot(loss):
    min_loss = int(np.where(np.min(loss) == loss)[0][0])

    plt.plot(np.arange(len(loss)), loss, label="Loss")
    plt.plot(
        np.arange(len(loss))[min_loss],
        loss[min_loss],
        "ro",
        label=f"Min loss {loss[min_loss]:.3e}",
    )
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.title("MSE as function of epochs")
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    plt.legend()
    plt.show()
    
def exact_solution(x, t, i):
    return np.sin(np.pi * x) * np.exp(-np.pi ** 2 * t[i])



def rel_err(data, x, t, index):
    rel_err_ = np.abs((data[index][1:-1] - exact_solution(x, t, index)[1:-1])/exact_solution(x, t, index)[1:-1])
    return rel_err_

def rel_err_total(data, x, t):
    error = np.zeros(len(t))

    for index, time in enumerate(t):
        rel_err_ = rel_err(data, x, t, index)
        error[index] = np.mean(rel_err_)

    return error
