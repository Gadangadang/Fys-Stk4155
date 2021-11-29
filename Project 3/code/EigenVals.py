from ML_PDE_solver import *
from numpy import linalg as LA


class EigenVal(NeuralNetworkPDE):
    def __init__(self, t, epochs, lr, A):
        self.t = tf.cast(tf.convert_to_tensor(t), tf.float32)
        self.X_0 = self.get_init_state()
        self.XX_0 = tf.transpose(self.X_0) @ self.X_0
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
        X = tf.reshape(X,[6,1])
        X_T = tf.transpose(X)

        lmb = (X_T @ self.A@X) / (X_T @ X)
        return lmb

    @tf.function
    def cost_function(self, model):
        with tf.GradientTape() as tape:
            t = self.t
            tape.watch(t)
            X =  model(t, training=True)
        #X = tf.reshape(X,[6,len(X)])
        X_dt = tape.gradient(X,t)
        del tape
        X_T = tf.transpose(X)
        error = 0
        for i in range(len(t)):
            X_i = X[i]
            AX = tf.linalg.matvec(tf.cast(A,tf.float32), X_i)
            f = tf.linalg.matvec(self.XX_0, AX) + (X_T[:,i]*AX)*X_i
            error += f  - X_dt[i]
        #print((X_T @ self.A@X ))
        #exit()
        #f = self.XX_0@self.A@X_T + X_T @ self.A@X @self.I@X_T

        MSE = tf.square(error)/len(t)
        print(MSE)

        #residual = f  - X_dt
        #MSE = tf.reduce_mean(tf.square(residual))
        return MSE

    def get_init_state(self):
        X_0 =  tf.cast(tf.convert_to_tensor(np.random.rand(1,6)), tf.float32)
        return X_0

    def track_EigenVal(self,loss):
        self.process[0].append(loss)
        self.process[1].append(self())



if __name__ == "__main__":
    a = np.random.rand(1,6)
    a_T = a.reshape(6,1)
    A = a_T@a
    Q = np.random.rand(6,6)
    A = (Q.T + Q)/2

    T = 10000
    Nt = 100
    t = np.linspace(0,T, Nt).reshape(Nt,1)
    epochs = 1000
    lr = 0.01

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
