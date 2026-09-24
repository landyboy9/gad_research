import numpy as np
from config import SEED, d, GAMMA, ZETA, R_VALUES, KAPPA_VALUES, p, num_iter
from utils import make_beta_gamma, generate_data, plot, prox_ell, ell_prime, ell_double_prime, sigmoid, s_func_poly, sigma_func_poly, oracle_risk_mc
from sklearn.linear_model import LogisticRegression
import warnings
from sklearn.exceptions import ConvergenceWarning
from scipy.optimize import least_squares, minimize_scalar, brentq


def ml_residuals(vars, H, G, Y, kappa):
    mu, alpha, lam = vars
    arg = alpha * H + mu * Y * G
    v = prox_ell(arg, lam)

    eq1 = np.mean(Y * G * v) - mu
    lp = ell_prime(v)
    eq2 = (lam**2) * np.mean(lp**2) - (alpha**2) * kappa
    ldp = ell_double_prime(v)
    eq3 = lam * np.mean(ldp / (1 + lam * ldp)) - kappa
    return [eq1, eq2, eq3]

def solve_ml(H, G, Y, kappa, x0=(0.5, 1.0, 1.0)):
    sol = least_squares(
        ml_residuals, x0=x0, args=(H, G, Y, kappa),
        bounds=([-np.inf, 1e-6, 1e-6], [np.inf, np.inf, np.inf])  # alpha, lam > 0
    )
    return sol.x   # mu, alpha, lam

def ml_risk(H, G, Y, kappa, x0=(0.5,1.0,1.0)):
    mu, alpha, lam = solve_ml(H, G, Y, kappa, x0)
    risk = np.mean((mu * G * Y + alpha * H) < 0)
    return risk, (mu, alpha, lam)

def eta_fn(q, rho, H, G, Y, kappa):
    val = np.minimum(rho*G*Y + np.sqrt(1-rho**2)*H - 1/q, 0)**2
    return np.mean(val) - (1 - rho**2) * kappa

def inner_min_over_rho(q, H, G, Y, kappa):
    res = minimize_scalar(lambda rho: eta_fn(q, rho, H, G, Y, kappa),
                           bounds=(-1, 1), method='bounded')
    return res.fun, res.x

def solve_svm(H, G, Y, kappa, q_lo=1e-2, q_hi=1e3):
    def outer_target(q):
        val, _ = inner_min_over_rho(q, H, G, Y, kappa)
        return val
    # bracket check
    lo_val, hi_val = outer_target(q_lo), outer_target(q_hi)
    if lo_val * hi_val > 0:
        # widen or grid-search for a sign change
        grid = np.geomspace(q_lo, q_hi, 40)
        vals = [outer_target(q) for q in grid]
        for i in range(len(grid)-1):
            if vals[i]*vals[i+1] < 0:
                q_lo, q_hi = grid[i], grid[i+1]
                break
        else:
            raise ValueError("No sign change found for q — widen search range")
    q_star = brentq(outer_target, q_lo, q_hi)
    _, rho_star = inner_min_over_rho(q_star, H, G, Y, kappa)
    return q_star, rho_star

def svm_risk(H, G, Y, kappa):
    q_star, rho_star = solve_svm(H, G, Y, kappa)
    risk = np.mean((rho_star*G*Y + np.sqrt(1-rho_star**2)*H) < 0)
    return risk, (q_star, rho_star)

def theoretical_risk(kappa, kappa_star, s, sigma, M, rng, warm_start_ml=(0.5,1.0,1.0)):
    H = rng.standard_normal(M)
    G = rng.standard_normal(M)
    Z = rng.standard_normal(M)
    p_y = sigmoid(s*G + sigma*Z)
    Y = 2*rng.binomial(1, p_y, size=M) - 1

    if kappa < kappa_star:
        risk, params = ml_risk(H, G, Y, kappa, x0=warm_start_ml)
    else:
        risk, params = svm_risk(H, G, Y, kappa)
    return risk, params

def g_of_kappa(kappa, r, gamma_exp, M, rng):
    s = s_func_poly(kappa, r, gamma_exp)
    sigma = sigma_func_poly(kappa, r, gamma_exp)

    H = rng.standard_normal(M)
    G = rng.standard_normal(M)
    Z = rng.standard_normal(M)
    p_y = sigmoid(s * G + sigma * Z)   # this is your existing sigmoid function
    Y = 2 * rng.binomial(1, p_y, size=M) - 1

    def objective(t):
        return np.mean(np.minimum(H + t * G * Y, 0) ** 2)

    res = minimize_scalar(objective, bounds=(-20, 20), method='bounded')
    return res.fun

def find_kappa_star(r, gamma_exp, zeta, M=100_000, seed=0):
    rng = np.random.default_rng(seed)

    def h(kappa):
        return g_of_kappa(kappa, r, gamma_exp, M, rng) - kappa

    kappa_grid = np.linspace(1e-3, zeta, 50)
    h_vals = [h(k) for k in kappa_grid]

    for i in range(len(kappa_grid) - 1):
        if h_vals[i] * h_vals[i + 1] < 0:
            return brentq(h, kappa_grid[i], kappa_grid[i + 1])

    raise ValueError("No sign change found — check bracket range or model parameters")

rng = np.random.default_rng(SEED)
big_train_errors = np.zeros((len(R_VALUES), len(KAPPA_VALUES)))
big_test_errors = np.zeros((len(R_VALUES), len(KAPPA_VALUES)))

n_values = []
d_vals = []

for r in range(len(R_VALUES)):
    oracle_risks = {r_val: oracle_risk_mc(r_val, M=200_000, rng=rng) for r_val in R_VALUES}
    for k in range(len(KAPPA_VALUES)):
        n = int(p/KAPPA_VALUES[k])
        n_values.append(n)
        d_k = int(ZETA*n)
        d_vals.append(d_k)
        # beta, gamma = make_beta_gamma(p, d_k, KAPPA_VALUES[k], R_VALUES[r], GAMMA, rng)
        train_errors = np.zeros(num_iter)
        test_errors = np.zeros(num_iter)
        for t in range(num_iter):
            beta, gamma = make_beta_gamma(p, d_k, KAPPA_VALUES[k], R_VALUES[r], GAMMA, rng)
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
            excess_risk = test_err - np.mean(y_test != np.sign(X_test @ np.hstack((beta, gamma))))

            excess_risk = test_err - oracle_risks[R_VALUES[r]]
            # test_errors[t] = excess_risk
            test_errors[t] = test_err

        big_train_errors[r, k] = np.mean(train_errors)
        big_test_errors[r, k] = np.mean(test_errors)


theory_risks = np.zeros((len(R_VALUES), len(KAPPA_VALUES)))
kappa_stars = np.zeros(len(R_VALUES))
ml_x0 = (0.5, 1.0, 1.0)   # warm-start carried across kappa

for r_idx, r in enumerate(R_VALUES):
    kappa_star = find_kappa_star(r, GAMMA, ZETA)
    kappa_stars[r_idx] = kappa_star
    ml_x0 = (0.5, 1.0, 1.0)  # reset warm start per r-curve
    for k_idx, kappa in enumerate(KAPPA_VALUES):
        s = s_func_poly(kappa, r, GAMMA)
        sigma = sigma_func_poly(kappa, r, GAMMA)
        risk, params = theoretical_risk(kappa, kappa_star, s, sigma, M=100_000, rng=rng, warm_start_ml=ml_x0)
        if kappa < kappa_star:
            ml_x0 = params  # warm-start next kappa's ML solve
        theory_risks[r_idx, k_idx] = risk

plot(big_train_errors, big_test_errors, theory_risks, kappa_stars)
