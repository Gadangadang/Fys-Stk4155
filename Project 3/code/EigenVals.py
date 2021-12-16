from ML_PDE_solver import *
from numpy import linalg as LA
tf.config.run_functions_eagerly(True)


class EigenVal(NeuralNetworkPDE):
    """
    Eigenvalue solver class

    Args:
        NeuralNetworkPDE (class): superclass for the eigenvalue class
    """
    def __init__(self, t, epochs, lr, A):
        """
        Initializes the eigenvalue solver object.

        Args:
            t (ndarray): time array
            epochs (int): number of epochs for training
            lr (float): learning rate for training
            A (ndarray): matrix to find the eigenvalues
        """
        self.t = tf.cast(tf.convert_to_tensor(t), tf.float32)
        self.X_0 = self.get_init_state()
        self.XX_0 = tf.reduce_sum(tf.transpose(self.X_0) * self.X_0)
        self.A = tf.cast(tf.convert_to_tensor(A), tf.float32)

        self.I = tf.eye(6)
        self.num_epochs = epochs
        self.learning_rate = lr
        self.in_out = [1, 6]

        self.tracker = self.track_EigenVal
        self.process = [[], [], []]

    def __call__(self):
        """
        Call function for object, to find the eigenvalue

        Returns:
            float: eigenvalue
        """
        X = self.model(tf.reshape(t[-1],[1,1]))
        X_T = tf.transpose(X)
        X_T_X = tf.reduce_sum(X_T * X)
        AX = tf.linalg.matvec(tf.cast(self.A,tf.float32), X)
        lmb =  tf.reduce_sum(X_T * AX) / (X_T_X)
        return lmb

    @tf.function
    def cost_function(self, model):
        """
        cost function for the network

        Args:
            model ([tensorflow_object): current model

        Returns:
            tensor: MSE tensor to calculate gradient
        """
        with tf.GradientTape() as tape:
            t = self.t
            tape.watch(t)
            X = model(t, training=True)
        X_dt = tape.gradient(X, t)
        del tape
        A = self.A
        X_T = tf.transpose(X)
        AX = tf.einsum("ij,kj->ki", A, X)
        LS = self.XX_0 * AX
        X_T_AX = tf.einsum("jk,kj -> k", X_T, AX)
        RS = tf.einsum("k,kj -> kj", X_T_AX, X)
        MSE = tf.reduce_mean((LS - RS - X_dt)**2, 0)
        return MSE

    def get_init_state(self):
        """
        Calculate initial state

        Returns:
            tensor: tensor of the initial state
        """
        X_0 = tf.cast(tf.convert_to_tensor(np.random.rand(1, 6)), tf.float32)
        return X_0

    def track_EigenVal(self, loss):
        """
        Tracks the eigenvalues

        Args:
            loss (tensor): loss tensor
        """
        lmb = self()
        vec = self.model(tf.reshape(t[-1],[1,1]))
        vec = vec/tf.norm(vec)
        self.print_string = f"Lambda = {lmb:.2e}"
        self.process[0].append(loss)
        self.process[1].append(lmb)
        self.process[2].append(vec)


if __name__ == "__main__":
    np_seed = 2
    np.random.seed(np_seed)
    Q = np.random.rand(6, 6)
    seed = 5
    np.random.seed(seed)
    tf.random.set_seed(seed)
    A = (Q.T + Q) / 2
    w, v = LA.eig(A)
    T = 1e6
    Nt = 100
    t = np.linspace(0, T, Nt).reshape(Nt, 1)
    epochs = 2000
    lr = 5e-4#1e-31e-4
    EV = EigenVal(t, epochs,  lr, A)
    loss, lmbds, EigenVec = EV.train()
    EigenVec = np.asarray(EigenVec).reshape((len(EigenVec),6))
    loss = np.asarray(loss).ravel()
    lmbds = np.asarray(lmbds).ravel()
    x = np.linspace(0, len(lmbds)-1, 10)
    indx = np.where(abs(w - lmbds[-1]) == np.min(abs(w - lmbds[-1]) ))

    """ Plot of Eigen value """
    plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
    plt.plot(np.arange(len(lmbds)), lmbds, label = "Neural network")
    for i in range(6):
        if w[i] == w[indx[0][0]]:
            i_max = i
            print(f"Error: {abs(w[i]-lmbds[-1]):.1e}")
            plt.plot(x, w[i] * np.ones(len(x)), "--", label = "Num-Diag")
        else:
            plt.plot(x, w[i] * np.ones(len(x)), "--")
    plt.xlabel("# Epochs", fontsize=16)
    plt.ylim([-1, 1])
    plt.ylabel("Value", fontsize=16)
    plt.title("Eigenvalue" + r" $[\lambda$]", fontsize = 16)
    plt.legend(fontsize=16)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    plt.savefig("../article/figures/NNEigVals.pdf", bbox_inches="tight")
    plt.show()
    print(EigenVec[-1,:])
    print(v[:,indx[0][0]])


    """ Plot of Eigen vectors """
    plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
    for i in range(6):
        label_str = f"i={i} "
        plt.plot(np.arange(len(EigenVec)), -EigenVec[:,i], label = label_str)
    plt.xlabel("# Epochs", fontsize=16)
    plt.ylabel("Value", fontsize=16)
    plt.legend(loc = "lower right", fontsize = 14)
    plt.title("Eigenvectors"+r" [$v$]", fontsize = 14)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    plt.savefig("../article/figures/EigenVec.pdf", bbox_inches="tight")
    plt.show()

    """ Plot of Eigen vs Diag solution """
    plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
    plt.plot(np.arange(6), EigenVec[-1,:], label = "Neural network: "+r"$\langle v \rangle$ = " + f"{abs(np.mean(EigenVec[-1,:])):.3f}")
    plt.plot(np.arange(6), -v[:,indx[0][0]],"--", label = "Num-Diag: "+r"$\langle v \rangle$ = " + f"{abs(np.mean(v[:,i_max])):.3f}")
    plt.xlabel("Index (i)", fontsize=16)
    plt.ylabel("Value", fontsize=16)
    plt.legend(loc = "upper right", fontsize = 14)
    plt.title("Eigenvectors: Neurlal network vs Num-Diag "+"[$v$]", fontsize = 16)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    plt.savefig("../article/figures/NNvsDiagVec.pdf", bbox_inches="tight")
    plt.show()
