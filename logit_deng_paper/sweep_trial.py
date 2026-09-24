import numpy as np
from config import SEED, d, GAMMA, ZETA, R_VALUES, KAPPA_VALUES, p, num_iter, eta_norm, big_sample_size
from utils import make_beta_gamma, generate_data, plot, s_func_poly, sigma_func_poly, oracle_risk_mc, sigmoid, sample_training_data
from sklearn.linear_model import LogisticRegression
import warnings
from sklearn.exceptions import ConvergenceWarning
from solvers import find_kappa_star, theoretical_risk

if p>=d:
    raise ValueError("p must be less than d for this experiment")

rng = np.random.default_rng(SEED)

# Create big boy data set
eta = rng.standard_normal(d)
eta = eta_norm * eta / np.linalg.norm(eta)
big_X = rng.standard_normal((big_sample_size, d))
big_y = 2*np.round(sigmoid(big_X@eta))-1
big_y = 2 * rng.binomial(1, sigmoid(big_X @ eta)) - 1

n_vals = (p/KAPPA_VALUES).astype(int)
if np.any(n_vals<=0):
    raise ValueError('Got n-val less than or equal to zero')

use_these_K_N = [(KAPPA_VALUES[k_idx], n_vals[k_idx]) for k_idx in range(len(KAPPA_VALUES)) if n_vals[k_idx] > 0]
big_train_errors = np.zeros((len(R_VALUES),len(use_these_K_N)))
big_test_errors = np.zeros((len(R_VALUES),len(use_these_K_N)))

for r_idx, r in enumerate(R_VALUES):
    for idx, pair in enumerate(use_these_K_N):
        k, n = pair
        train_errors = np.zeros(num_iter)
        test_errors = np.zeros(num_iter)

        for t in range(num_iter):
            print(t)
            
            W_train, y_train, W_test, y_test, row_idx, col_idx = sample_training_data(big_X, big_y, p, n, rng)

            # Train using logistic regression with no regularization
            clf = LogisticRegression(C=np.inf, fit_intercept=False, max_iter=10_000, tol=1e-6)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                clf.fit(W_train, y_train)

            beta_hat = clf.coef_.flatten()

            # Calculate training error
            train_pred = np.sign(W_train @ beta_hat)
            train_err = np.mean(train_pred != y_train)
            train_errors[t] = train_err

            # Test error
            test_pred = np.sign(W_test @ beta_hat)
            test_err = np.mean(test_pred != y_test)
            test_errors[t] = test_err

        big_train_errors[r_idx,idx] = np.mean(train_errors)
        big_test_errors[r_idx,idx] = np.mean(test_errors)


   

plot(big_train_errors, big_test_errors, None, None)
