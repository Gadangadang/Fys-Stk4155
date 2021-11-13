import autograd.numpy as np

t_model = np.array([0.15, 0.2]).reshape(2,1)
target = np.array([0.2, 0.21]).reshape(2,1)

print(ydata)

val = 1-np.sum((ydata - ymod)**2) /\
       np.sum((ydata - np.mean(ydata, axis=0) ** 2))

print(val)



def R2_score(self, X, target):
    ymod = self.predict(X)
    target = target

    return 1-np.sum((target - ymod)**2) /\
           np.sum((target - np.mean(target, axis=0) ** 2))
