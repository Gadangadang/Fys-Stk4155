import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

cwd = os.getcwd() # current working directory
data_path = cwd + '/Ames_Housing_dataset'
df = pd.read_csv (data_path + '/train.csv')


features = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'YearRemodAdd', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold',]
target = ['SalePrice']


def fetch_data(top_features = 10, split = 0.2):
    """
    Fetch the features with top correlation
    """
    np.random.seed(4155)
    correlation = df[df.columns[1:]].corr()['SalePrice'][features]
    corr = correlation.to_numpy()
    order = np.argsort(corr)
    top = order[-top_features:]
    top_features = [features[i] for i in top]

    X = df[top_features].to_numpy()
    y = df[target].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split)
    X_train, X_test, y_train, y_test = scale_MinMax(X_train, X_test, y_train, y_test)
    return X_train, X_test, y_train, y_test

def scale_MinMax(*args):
    scaled = []
    scaler = MinMaxScaler()
    for arg in args:
        scaler.fit(arg)
        scaled.append(scaler.transform(arg))

    if len(args) == 1:  # If just one argument
        return scaled[0]
    else:
        return scaled

def cal_MSE(y_data, y_model):
    return np.mean((y_data.ravel() - y_model.ravel())**2)

def cal_MSE_rel(y_data, y_model):
    return np.mean(((y_data.ravel() - y_model.ravel())/y_data.ravel())**2)


def R2(y_data, y_model):
    return 1 - np.sum((y_data.ravel() - y_model.ravel())**2) / np.sum((y_data.ravel() - np.mean(y_data.ravel())) ** 2)



def watch_data():
    for i in range(X_train.shape[1]):
        plt.plot(X_train[:,i], target, 'o')
        plt.xlabel(features[i])
        plt.show()


if __name__ == "__main__":


    X, y = fetch_data(10)
    # watch_data()
