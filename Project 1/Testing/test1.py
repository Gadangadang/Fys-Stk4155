import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)
def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n


x = np.random.rand(100)
y = 2.0+5*x*x+0.1*np.random.randn(100)

X = np.zeros((len(x),3))
X[:,0] = 1.0
X[:,1] = x
X[:,2] = x**2

Beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
ytilde = X @ Beta

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

# and then make the prediction
ytildenp = X_train @ beta

print("Training R2")
print(R2(y_train,ytildenp))
print("Training MSE")
print(MSE(y_train,ytildenp))

ypredict = X_test @ beta

print("Test R2")
print(R2(y_test,ypredict))
print("Test MSE")
print(MSE(y_test,ypredict))


plt.scatter(x, y,label="Data")
plt.scatter(x, ytilde, label="Lin Reg")
#plt.scatter(X_test, y_test, label="Lin Reg 2")
plt.legend()
plt.show()
