from bias_variance_analysis import *


X_train, X_test, y_train, y_test = fetch_data()
B = 100

bv_object = bias_variance(X_train, X_test, y_train, y_test)

n_arr = np.linspace(1, 100, 10).astype("int")
print(n_arr)
exit()
layers = [1, 3, 5, 10]
for layers_ in layers:
    bv_object.set_method("NN_variable_nodes", n_arr, 3)
    bv_object.analyze(num_bootstraps = B)
    bv_object.plot(show = False, save = f"NN_variable_nodes_{layers_}_{B}")

B = 100
n_arr = np.linspace(1, 20, 10).astype("int")
nodes = [5, 20, 50, 100]
for nodes_ in nodes:
    bv_object.set_method("NN_variable_layers", n_arr, 3)
    bv_object.analyze(num_bootstraps = B)
    bv_object.plot(show = False, save = f"NN_variable_layers_{nodes_}_{B}")
