import torch
import torchvision
from torchvision.datasets import MNIST

if __name__ == "__main__":


    #run for logistic regression.
    solver = SGD(X_train, Z_train, eta_val=0.001, m = 100, num_epochs = int(1e4), gradient_func = "Logistic")

    theta_SGD = solver.SGD_train()

    print(f"{solver.accuracy_score(X_test,Z_test)*100:.0f}%")
