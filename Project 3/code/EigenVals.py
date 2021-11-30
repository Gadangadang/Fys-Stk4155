from ML_PDE_solver import *
from numpy import linalg as LA
tf.config.run_functions_eagerly(True)

class EigenVal(NeuralNetworkPDE):
    def __init__(self, t, epochs, lr, A):
        self.t = tf.cast(tf.convert_to_tensor(t), tf.float32)
        self.X_0 = self.get_init_state()
        self.XX_0 = tf.reduce_sum(tf.transpose(self.X_0) * self.X_0)
        self.A = tf.cast(tf.convert_to_tensor(A), tf.float32)

        self.I = tf.eye(6)
        self.num_epochs = epochs
        self.learning_rate = lr
        self.in_out = [1,6]

        self.g_t_jacobian_func = jacobian(self.g_trial, 0)
        self.g_t_hessian_func = hessian(self.g_trial, 0)

        self.tracker = self.track_EigenVal
        self.process = [[],[]]

    def __call__(self):
        X = self.model(self.t)[-1]
        X_T = tf.transpose(X)
        X_T_X = tf.reduce_sum(X_T * X)
        AX = tf.linalg.matvec(tf.cast(A,tf.float32), X)
        lmb =  tf.reduce_sum(X_T * AX) / (X_T_X)
        return lmb

    @tf.function
    def cost_function(self, model):
        with tf.GradientTape() as tape:
            t = self.t
            tape.watch(t)
            X =  model(t, training=True)
        X_dt = tape.gradient(X,t)
        del tape

        X_T = tf.transpose(X)
        AX = tf.einsum("ij,kj->ki", A,X)
        LS = self.XX_0 *AX
        X_T_AX = tf.einsum("jk,kj -> k", X_T,AX)
        RS = tf.einsum("k,kj -> kj", X_T_AX,X)

        MSE = tf.reduce_mean(LS + RS - X_dt,0)
        return MSE

    def get_init_state(self):
        X_0 =  tf.cast(tf.convert_to_tensor(np.random.rand(1,6)), tf.float32)
        return X_0

    def track_EigenVal(self,loss):
        lmb = self()
        self.print_string = f"Lambda = {lmb:.2f}"
        self.process[0].append(loss)
        self.process[1].append(lmb)



if __name__ == "__main__":

    Q = np.random.rand(6,6)
    A = (Q.T + Q)/2

    T = 1e30
    Nt = 100
    t = np.linspace(0,T, Nt).reshape(Nt,1)
    epochs = 10000
    lr = 0.0001
    #check out einsum.

    EV = EigenVal(t, epochs,  lr, A)
    loss, lmbds = EV.train()
    loss = np.asarray(loss).ravel()
    lmbds = np.asarray(lmbds).ravel()
    w, v = LA.eig(A)

    plt.plot(np.arange(len(lmbds)), lmbds)
    x = np.linspace(0, len(lmbds), 10)

    for i in range(6):
        plt.plot(x, w[i]*np.ones(len(x)), "--")
    plt.show()
    plt.plot(np.arange(len(loss)), loss)
