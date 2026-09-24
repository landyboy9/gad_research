import numpy as np
import config
import matplotlib.pyplot as plt
import config

# Define basic logistic function
def sigmoid(t):
    return 1/(1+np.exp(-t))

def ell_prime(t):      # derivative of log(1+e^{-t})
    return -1 / (1 + np.exp(t))          # = sigmoid(t) - 1, always in (-1, 0)

def ell_double_prime(t):   # second derivative
    s = sigmoid(t)
    return s * (1 - s)                   # always positive

def prox_ell(x, lam, n_iter=50):
    v = x.copy()
    for _ in range(n_iter):
        F = v - x + lam * ell_prime(v)
        Fp = 1 + lam * ell_double_prime(v)
        v = v - F / Fp
    return v

# Randomly generate eta vector (the actual coefficients for the logistic model)
def make_beta_gamma(p, d, kappa, r, gamma_exp, rng):
    s2 = r**2 * (1 - (1+kappa)**(-gamma_exp))   # polynomial model, eq (12)
    s = np.sqrt(s2)
    sigma = np.sqrt(r**2 - s2)

    beta0_dir = rng.standard_normal(p)
    beta0 = s * beta0_dir / np.linalg.norm(beta0_dir)

    gamma0_dir = rng.standard_normal(d - p)
    gamma0 = sigma * gamma0_dir / np.linalg.norm(gamma0_dir)

    return beta0, gamma0


def make_beta_gamma_linear(p, d, kappa, r, rng):
    if kappa>config.ZETA:
        raise ValueError("Kappa must be between zero and ZETA")
    s2 = r**2 * (kappa/config.ZETA)   # polynomial model, eq (12)
    s = np.sqrt(s2)
    sigma = np.sqrt(r**2 - s2)

    beta0_dir = rng.standard_normal(p)
    beta0 = s * beta0_dir / np.linalg.norm(beta0_dir)

    gamma0_dir = rng.standard_normal(d - p)
    gamma0 = sigma * gamma0_dir / np.linalg.norm(gamma0_dir)

    return beta0, gamma0

# Generate synthetic data with n samples
def generate_data(n, p, d, beta0, gamma0, rng):
    W = rng.standard_normal((n, p))       # known features
    Zf = rng.standard_normal((n, d - p))  # unknown features
    X = np.hstack((W, Zf))
    logits = W @ beta0 + Zf @ gamma0
    y = 2 * rng.binomial(1, sigmoid(logits)) - 1
    return X, W, y   # only W (not Zf) is given to the learner

def plot(train_data, test_data, theory_risks, kappa_stars):
    plt.rcParams.update({'font.family': 'serif','mathtext.fontset': 'cm',})
    colors = ['g','r','b','c','m','y','k']
    for i in range(len(config.R_VALUES)):
        if train_data is not None:
            if theory_risks is None:
                plt.plot(config.KAPPA_VALUES, train_data[i, :], marker='o',color=colors[i], label="Train")
            else:
                plt.scatter(config.KAPPA_VALUES, train_data[i, :], marker='o',color=colors[i])
        if test_data is not None:
            if theory_risks is None:
                plt.plot(config.KAPPA_VALUES, test_data[i,:], marker='x',color=colors[i],label='Test')
            else:
                plt.scatter(config.KAPPA_VALUES, test_data[i, :], marker='x',color=colors[i])
        if theory_risks is not None:
            plt.plot(config.KAPPA_VALUES, theory_risks[i, :], label=rf"$r={config.R_VALUES[i]}$", color=colors[i])
        if kappa_stars is not None:
            plt.axvline(x=kappa_stars[i], color=colors[i], linestyle='--', alpha=0.5)
    plt.xlabel(r"Overparametrization ratio $\kappa$")
    plt.ylabel("Zero-One Error")
    plt.legend()
    if config.FIX_PLOT_SCALE:
        plt.xlim(0, 3)
        plt.ylim(0,.5)
    if config.PLOT_TITLE is not None:
        plt.title(config.PLOT_TITLE)
    plt.tight_layout()
    plt.savefig(f'{config.FILE_STRING}.png', dpi=400)
    plt.show()

def s_func_poly(kappa, r, gamma_exp):
    s2 = r**2 * (1 - (1 + kappa) ** (-gamma_exp))
    return np.sqrt(s2)

def sigma_func_poly(kappa, r, gamma_exp):
    s = s_func_poly(kappa, r, gamma_exp)
    return np.sqrt(r**2 - s**2)

def oracle_risk_mc(r, M=200_000, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    u = rng.standard_normal(M) * r
    return np.mean(sigmoid(-np.abs(u)))

def plot_heat(matrix):
    plt.rcParams.update({'font.family': 'serif','mathtext.fontset': 'cm',})
    plt.pcolormesh(config.n_values, config.p_values, matrix)
    plt.colorbar(label='Test Error')
    plt.xlabel(r'Training Sample Size ($n$)')
    plt.ylabel(r'Number of paraemeters ($p$)')
    if config.PLOT_TITLE is not None:
        plt.title(config.PLOT_TITLE)
    plt.tight_layout()
    plt.savefig(f'deng_heatmap{config.FILE_STRING}.png', dpi=400)
    plt.show()

def sample_training_data(X, y, n_params, n_datapoints, rng):
    
    n_samples, n_features = X.shape

    if n_params > n_features:
        raise ValueError(f"n_params ({n_params}) > available features ({n_features})")
    if n_datapoints > n_samples:
        raise ValueError(f"n_datapoints ({n_datapoints}) > available samples ({n_samples})")

    row_idx = rng.choice(n_samples, size=n_datapoints, replace=False)
    col_idx = rng.choice(n_features, size=n_params, replace=False)

    # rows NOT selected for training -> test set
    mask = np.ones(n_samples, dtype=bool)
    mask[row_idx] = False
    test_row_idx = np.nonzero(mask)[0]

    X_train = X[np.ix_(row_idx, col_idx)]
    y_train = y[row_idx]

    X_test = X[np.ix_(test_row_idx, col_idx)]
    y_test = y[test_row_idx]

    return X_train, y_train, X_test, y_test, row_idx, col_idx