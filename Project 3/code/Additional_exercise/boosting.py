from bias_variance_analysis import *


X_train, X_test, y_train, y_test = fetch_data()
n_arr = np.arange(1, 20)
B = 1000
bv_object = bias_variance(X_train, X_test, y_train, y_test)
tree_depth = [1, 2, 3, 4]
for depth in tree_depth:
    bv_object.set_method("boosting", n_arr, 3)
    bv_object.analyze(num_bootstraps = B)
    bv_object.plot(show = False, save = f"boosting_{depth}_{B}")
