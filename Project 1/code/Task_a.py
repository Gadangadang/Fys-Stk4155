import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from Functions import *
from plot_set import *
from matplotlib.ticker import MaxNLocator


def confidence_interval(beta, X):
    """Finds the confidence interval for beta values

    Args:
        beta (Array): Array containing all the beta values
        X    (Array): Design matrix

    Returns:
        Array: Interval array
        String: print string
        Float: Uncertainty for a given beta value
    """
    alpha = 0.95
    Z = norm.ppf(alpha + (1 - alpha) / 2)  # Calculate Z
    beta_var = np.linalg.pinv(X.T @ X).diagonal()  # Find the variance
    SE_i = np.sqrt(beta_var)  # Find standard error
    # Zip the interval.
    conf_int = np.dstack((beta - Z * SE_i, beta + Z * SE_i))[0]
    uncertainty = Z * SE_i

    uncertainty_print = f"Beta    Uncertainty \n"
    for i in range(len(beta)):
        uncertainty_print += f"{beta[i]:4.2g} +- {uncertainty[i]:2.1g}\n"

    return conf_int, uncertainty_print, uncertainty


def latex_table(beta, pm_train, pm_test):
    """Creates table for latex

    Args:
        beta (Array): Array with beta values
        pm_train ([type]): Uncertainty for train data
        pm_test ([type]): Uncertainty for train data
    """
    print(r"\begin{table}[H]")
    print(r"\begin{center}")
    print(r"\caption{$\beta$-values for OLS regression with a polynomial up to degree five using $N = 25$ and $\sigma = 0.2$. The uncertainty is the result of a $95\%$-confidence interval.}")
    print(r"\begin{tabular}{|c|c|c|} \hline")
    print(
        r" \text{Beta} & \text{Train uncertainty} & \text{Test uncertainty} \\\hline")
    for i in range(len(pm_train)):
        print(
            f"{beta[i]:2.2f} & $\pm${pm_train[i]:.2f} & $\pm${pm_test[i]:.2f}" + r"\\\hline")
    print(r"\end{tabular}")
    print(r"\label{tab:beta_uncertainty}")
    print(r"\end{center}")
    print(r"\end{table}")


def confidence_plot(conf_int_train, conf_int_test):
    """Confidence interval plot

    Args:
        conf_int_train (Array): Array with confidence intervals for train data
        conf_int_test (Array): Array with confidence intervals for test data
    """
    beta_train = np.mean(conf_int_train, axis=1)
    train_error = np.abs(conf_int_train[:, 0] - conf_int_train[:, 1]) / 2
    beta_test = np.mean(conf_int_test, axis=1)
    test_error = np.abs(conf_int_test[:, 0] - conf_int_test[:, 1]) / 2

    plt.figure(num=0, figsize=(8, 6), facecolor='w', edgecolor='k')
    b_arr = np.linspace(0, len(beta_train) - 1, len(beta_train))

    plt.subplot(2, 1, 1)
    plt.errorbar(b_arr, beta_train, yerr=train_error,
                 color=color_cycle(0), fmt='o', label="train")
    ax = plt.gca()
    # Force integer ticks on x-axis
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel(r"$i$", fontsize=14)
    plt.ylabel(r"$\hat{\beta}_i$", fontsize=14)
    plt.legend(fontsize=13)

    plt.subplot(2, 1, 2)
    plt.errorbar(b_arr, beta_test, yerr=test_error,
                 color=color_cycle(1), fmt='o', label="test")
    ax = plt.gca()
    # Force integer ticks on x-axis
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel(r"i", fontsize=14)
    plt.ylabel(r"$\hat{\beta}_i$", fontsize=14)
    plt.legend(fontsize=13)

    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    plt.subplots_adjust(hspace=0.3)
    #plt.savefig("../article/figures/confidence_interval.pdf",
    #            bbox_inches="tight")

    plt.show()


def evaluate_regression(beta, X_train, X_test, z_train, z_test):
    """Prints the uncertainty for beta values for the test and training set

    Args:
        beta    (Array): Array with beta values
        X_train (Array): Array with training data from the design matrix
        X_test  (Array): Array with test data from the design matrix
        z_train (Array): Array with training data from the actual data
        z_test  (Array): Array with test data from the actual data
    """
    # Prediction
    ztilde = (X_train @ beta).ravel()
    zpredict = (X_test @ beta).ravel()

    MSE_train = MSE(z_train, ztilde)
    MSE_test = MSE(z_test, zpredict)
    R2_train = R2(z_train, ztilde)
    R2_test = R2(z_test, zpredict)

    # alpha-% confidential interval (standard normal distribution)
    alpha = 0.95
    from scipy.stats import norm

    conf_int_train, beta_uncertainty_print_train, uncertainty_train = confidence_interval(
        beta, X_train)
    conf_int_test, beta_uncertainty_print_test, uncertainty_test = confidence_interval(
        beta, X_test)

    confidence_plot(conf_int_train, conf_int_test)
    latex_table(beta, uncertainty_train, uncertainty_test)

    #--- print result ---#
    print_results = True
    if print_results:
        print("#----- Error -----#")
        print("      train  |  test")
        print(f"MSE: {MSE_train:2.3f} | {MSE_test:2.3f}")
        print(f"R2 : {R2_train:2.3f} | {R2_test:2.3f}")

        print(f"\n#----- {alpha}% confidence intervals -----#")

        print(f"\n__Training-set__")
        # print(conf_int_train)
        print(beta_uncertainty_print_train)
        print(f"\n__Test-set__")
        # print(conf_int_test)
        print(beta_uncertainty_print_test)


if __name__ == "__main__":

    #--- settings ---#
    N = 30             # Number of points in each dimension
    z_noise = 0.2       # Added noise to the z-value
    n = 6               # Highest order of polynomial for X

    # Create data and set up design matrix
    x, y, z = generate_data(N, z_noise)
    X = create_X(x, y, n)

    # Split data into train and test data
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

    # Scale bu subtracting mean
    mean_scale(X_train, X_test, z_train, z_test)

    # OLS regression
    beta_OLS = OLS_regression(X_train, z_train)
    evaluate_regression(beta_OLS, X_train, X_test, z_train, z_test)
