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
        X = tf.reshape(X,[6])
        X_T_X = tf.einsum("j,j -> ", X, X)
        AX = tf.einsum("kj,j -> k", self.A, X)
        lmb =  tf.einsum("j,j -> ", X, AX) / (X_T_X)
        return lmb
    def get_model(self):
        """
        Initializes the model, setting up layers and compiles the model,
        prepping for training.

        Returns:
            tensorflow_object: compiled model
        """
        nodes = 50
        get_custom_objects().update({"abs_activation": self.abs_activation})
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    nodes, activation="sigmoid", input_shape=(self.in_out[0],)
                ),
                tf.keras.layers.Dense(nodes, activation="sigmoid"),
                tf.keras.layers.Dense(self.in_out[1]),
            ]
        )
        self.optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=self.optimizer)
        model.summary()
        return model

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
        X_T_AX = tf.einsum("jk,kj -> k", X_T, AX)
        RS = tf.einsum("k,kj -> kj", X_T_AX, X)
        LS = self.XX_0 * AX
        MSE = tf.reduce_mean(LS + RS - X_dt, 0)
        return MSE

    def get_init_state(self):
        """
        Calculate initial state

        Returns:
            tensor: tensor of the initial state
        """
        X_0 = tf.cast(tf.convert_to_tensor(np.random.randn(1, 6)), tf.float32)
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
    #np_seed = 2
    #tf_seed = 1000
    np_seed = 10
    tf_seed = np_seed
    np.random.seed(np_seed)
    tf.random.set_seed(tf_seed)
    Q = np.random.randn(6, 6)
    A = (Q.T + Q) / 2
    T = 1e4
    Nt = 1000
    t = np.linspace(0, T, Nt).reshape(Nt, 1)
    epochs = 2500
    lr = 1e-4

    EV = EigenVal(t, epochs,  lr, A)
    loss, lmbds, EigenVec = EV.train()
    EigenVec = np.asarray(EigenVec).reshape((len(EigenVec),6))
    loss = np.asarray(loss).ravel()
    lmbds = np.asarray(lmbds).ravel()
    w, v = LA.eig(A)
    x = np.linspace(0, len(lmbds), 10)


    """ Plot of Eigen value """
    plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
    plt.plot(np.arange(len(lmbds)), lmbds, label = "NN: "+ \
    r" $\lambda_{max}$" +f" = {lmbds[-1]:.2f}")
    for i in range(6):
        if w[i] == np.max(w):
            plt.plot(x, w[i] * np.ones(len(x)), "--", \
            label = "Num-Diag: "+ r"$\lambda_{max} = $"+f"{w[i]:.2f}")
            i_max = i
        else:
            plt.plot(x, w[i] * np.ones(len(x)), "--")
    plt.xlabel("# Epochs", fontsize=16)
    plt.ylabel("Value", fontsize=16)
    plt.title("Eigenvalue" + r" $[\lambda$]")
    plt.legend(loc = "center right", fontsize = 16)
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    #plt.savefig("../article/figures/NNEigVals.pdf", bbox_inches="tight")
    plt.show()

    """ Plot of Eigen vectors """
    plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
    for i in range(6):
        label_str = f"i={i} "
        plt.plot(np.arange(len(EigenVec)), EigenVec[:,i], label = label_str)
    plt.xlabel("# Epochs", fontsize=16)
    plt.ylabel("Value", fontsize=16)
    plt.legend(loc = "lower right", fontsize = 14)
    plt.title("Eigenvectors"+r" [$v$]")
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    #plt.savefig("../article/figures/EigenVec.pdf", bbox_inches="tight")
    plt.show()

    """ Plot of Eigen vs Diag solution """
    plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
    plt.plot(np.arange(6), -EigenVec[-1,:], label = "Neural network: "+r"$\langle v \rangle$ =" + f"{np.mean(-EigenVec[-1,:]):.3f}")
    plt.plot(np.arange(6), v[:,i_max], label = "Num-Diag: "+r"$\langle v \rangle$ =" + f"{np.mean(v[:,i_max]):.3f}")
    plt.xlabel("Index (i)", fontsize=16)
    plt.ylabel("Value", fontsize=16)
    plt.legend(loc = "upper right", fontsize = 14)
    plt.title("Eigenvectors: Neurlal network vs Num-Diag "+"[$v$]")
    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    #plt.savefig("../article/figures/NNvsDiagVec.pdf", bbox_inches="tight")
    plt.show()
