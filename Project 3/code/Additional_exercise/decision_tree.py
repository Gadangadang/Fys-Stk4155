from bias_variance_analysis import *


X_train, X_test, y_train, y_test = fetch_data()
n_arr = np.arange(1, 15)

bv_object = bias_variance(X_train, X_test, y_train, y_test)
bv_object.set_method("decision_tree", n_arr)
bv_object.analyze(num_bootstraps = 5000)
bv_object.plot(show = True, save = "decision_tree_5000")
