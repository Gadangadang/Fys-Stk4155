import matplotlib.pyplot as plt
import numpy as np



def loss_plot(loss, name = None):
    """
    Plot loss as function of epochs

    Args:
        loss (array): loss array
    """
    min_loss = int(np.where(np.min(loss) == loss)[0][0])
    plt.figure(num=1, dpi=80, facecolor='w', edgecolor='k')
    plt.plot(np.arange(len(loss)), loss, label="Loss")
    plt.plot(
        np.arange(len(loss))[min_loss],
        loss[min_loss],
        "ro",
        label=f"Min loss {loss[min_loss]:.3e}",
    )
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("MSE", fontsize=14)
    plt.yscale("log")
    # plt.title("MSE as function of epochs")
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    plt.legend()
    if name is not None:
        plt.savefig(
            f"../article/figures/{name}.pdf", bbox_inches="tight")
    plt.show()
