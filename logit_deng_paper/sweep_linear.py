import numpy as np
from config import SEED, d, GAMMA, ZETA, R_VALUES, KAPPA_VALUES, p, num_iter
from utils import make_beta_gamma_linear, generate_data, plot
from sklearn.linear_model import LogisticRegression
import warnings
from sklearn.exceptions import ConvergenceWarning



rng = np.random.default_rng(SEED)
big_train_errors = np.zeros((len(R_VALUES), len(KAPPA_VALUES)))
big_test_errors = np.zeros((len(R_VALUES), len(KAPPA_VALUES)))

n_values = []
d_vals = []

for r in range(len(R_VALUES)):
    for k in range(len(KAPPA_VALUES)):
        n = int(p/KAPPA_VALUES[k])
        n_values.append(n)
        d_k = int(ZETA*n)
        d_vals.append(d_k)
        # beta, gamma = make_beta_gamma_linear(p, d_k, KAPPA_VALUES[k], R_VALUES[r], rng)
        train_errors = np.zeros(num_iter)
        test_errors = np.zeros(num_iter)
        for t in range(num_iter):
            beta, gamma = make_beta_gamma_linear(p, d_k, KAPPA_VALUES[k], R_VALUES[r], rng)
            X_train, W_train, y_train = generate_data(n, p, d_k, beta, gamma, rng)

            # Train using logistic regression with no regularization
            clf = LogisticRegression(C=np.inf, fit_intercept=False, max_iter=10000, tol=1e-6)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                clf.fit(W_train, y_train)

            beta_hat = clf.coef_.flatten()
            beta_hat_dir = beta_hat / np.linalg.norm(beta_hat)



            # Calculate training error
            train_pred = np.sign(W_train @ beta_hat_dir)
            train_err = np.mean(train_pred != y_train)
            train_errors[t] = train_err

            # More data to test with
            X_test, W_test, y_test = generate_data(n, p, d_k, beta, gamma, rng)
            test_pred = np.sign(W_test @ beta_hat_dir)
            test_err = np.mean(test_pred != y_test)

            test_errors[t] = test_err

        big_train_errors[r, k] = np.mean(train_errors)
        big_test_errors[r, k] = np.mean(test_errors)


plot(big_train_errors, big_test_errors, None, None)
