from ML_PDE_solver import *


class EigenVal(NerualNetworkPDE):
    def __init__(self, t, epochs, I, lr, A, in_out = [2,1]):
        self.t = tf.cast(tf.convert_to_tensor(t), tf.float32)
        self.X_0 = self.get_init_state()
        self.A = A

        self.I = I
        self.num_epochs = epochs
        self.learning_rate = lr
        self.in_out = in_out

        self.g_t_jacobian_func = jacobian(self.g_trial, 0)
        self.g_t_hessian_func = hessian(self.g_trial, 0)

    def __call__(self):
        return 0

    @tf.function
    def cost_function(self, model):
        with tf.GradientTape() as tape:
            t = self.t
            tape.watch(t)
            X = self.model(t)
        X_t = tape.gradient(u,t)
        f = X.T@X*self.A + (1-X.T * self.A@X )@ X
        residual = f - u - u_t
        MSE = tf.reduce_mean(tf.square(residual))
        return MSE

    def get_init_state():
        X_0 =  tf.cast(tf.convert_to_tensor(a = np.random.rand(1,6)), tf.float32)
        return X_0


if __name__ == "__main__":
    a = np.random.rand(1,6)
    a_T = a.reshape(6,1)
    A = a_T@a

    T = 1000
    t = np.linspace(0,T, 10)

    epochs = 100
    lr = 100

    ML = NeuralNetworkPDE(t, epochs, I, lr)
    loss = ML.train()
