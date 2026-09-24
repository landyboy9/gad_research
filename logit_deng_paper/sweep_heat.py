import numpy as np
from config import SEED, d, GAMMA, ZETA, num_iter, p_values, n_values, r
from utils import make_beta_gamma, generate_data, plot_heat
from sklearn.linear_model import LogisticRegression
import warnings
from sklearn.exceptions import ConvergenceWarning


rng = np.random.default_rng(SEED)
big_test_errors = np.zeros((len(p_values), len(n_values)))

for p_idx, p in enumerate(p_values):
    for n_idx, n in enumerate(n_values):
        k = p/n
        beta, gamma = make_beta_gamma(p, d, k, r, GAMMA, rng)
        test_errors = np.zeros(num_iter)
        for t in range(num_iter):
            
            X_train, W_train, y_train = generate_data(n, p, d, beta, gamma, rng)

            # retry if degenerate (all same class)
            retries = 0
            while len(np.unique(y_train)) < 2 and retries < 20:
                X_train, W_train, y_train = generate_data(n, p, d, beta, gamma, rng)
                retries += 1
            if len(np.unique(y_train)) < 2:
                continue  # skip this trial if still degenerate after retries


            # Train using logistic regression with no regularization
            clf = LogisticRegression(C=np.inf, fit_intercept=False, max_iter=10000, tol=1e-6)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                clf.fit(W_train, y_train)

            beta_hat = clf.coef_.flatten()
            beta_hat_dir = beta_hat / np.linalg.norm(beta_hat)


            # More data to test with
            X_test, W_test, y_test = generate_data(n, p, d, beta, gamma, rng)
            test_pred = np.sign(W_test @ beta_hat_dir)
            test_err = np.mean(test_pred != y_test)
            excess_risk = test_err - np.mean(y_test != np.sign(X_test @ np.hstack((beta, gamma))))
            test_errors[t] = test_err

        big_test_errors[p_idx, n_idx] = np.mean(test_errors)


plot_heat(big_test_errors)