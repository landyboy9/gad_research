"""
Estimating d f/d n, d f/d p, d g/d n, d g/d p for logistic regression
======================================================================

f(n, p) = expected TEST error with n training points, p parameters
g(n, p) = expected TRAIN error with n training points, p parameters

Core idea
---------
f and g are expectations over the randomness in (a) which n points get
sampled as the training set and (b) how "p parameters" is realized as a
feature map. We estimate them by Monte Carlo: repeatedly draw a training
set / feature map, fit logistic regression, record train and test loss,
and average. Then we get the partial derivatives two ways:

  1. Finite differences on the noisy grid, using COMMON RANDOM NUMBERS
     (CRN) so that the same randomness is reused across nearby (n,p)
     points. This makes the noise in f(n,p+dp) and f(n,p-dp) highly
     correlated, so it mostly cancels in the difference -- this is the
     single most important trick for making these derivatives usable.

  2. A smoothed 2D surface (bivariate spline) fit separately on each
     side of the interpolation threshold p = n, differentiated
     analytically. This avoids amplifying Monte Carlo noise and
     respects the fact that f and g are NOT smooth across p = n
     (there's a real kink/singularity there for logistic regression --
     see Candes & Sur 2020 on the separability phase transition).

Two knobs control what "p parameters" means, since real datasets have a
fixed number of raw columns d:
  - feature_mode="subset": p <= d, take the first p raw columns
    (only explores the underparameterized side p <= d).
  - feature_mode="random_features": p can be ANY size (p << d or
    p >> d) via a random feature map. This is what lets you cross the
    interpolation threshold and see double descent, following the
    random-features-regression setup used in Mei & Montanari (2019)
    and the logistic analogues (Deng, Kammoun & Thrampoulidis 2019;
    Salehi, Abbasi & Hassibi 2019).
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.interpolate import SmoothBivariateSpline


# ----------------------------------------------------------------------
# 1. Feature construction (controls what "p" means)
# ----------------------------------------------------------------------

def make_features(X_raw, p, W_full=None, mode="random_features", rng=None):
    """
    Build a p-dimensional feature matrix from X_raw (n x d).

    To get valid common-random-numbers cancellation across DIFFERENT
    values of p, features are NESTED: the first p columns of the
    p_max-dimensional feature map are reused for any p <= p_max. Pass
    the same W_full in for every p you evaluate within one replicate.

    Returns (features, W_full) so W_full can be reused/extended.
    """
    n, d = X_raw.shape
    if mode == "subset":
        p = min(p, d)
        return X_raw[:, :p], W_full

    elif mode == "random_features":
        # random_features: phi(x) = relu(X_raw @ W), W is d x p_max,
        # nested so column j is the same regardless of how large p_max
        # was when W_full was generated for THIS replicate's rng seed.
        if W_full is None:
            raise ValueError("random_features mode requires a pre-generated W_full")
        p = min(p, W_full.shape[1])
        Z = X_raw @ W_full[:, :p] / np.sqrt(d)
        return np.maximum(Z, 0.0), W_full

    else:
        raise ValueError(f"unknown feature_mode {mode}")


def make_W_full(d, p_max, rng):
    """Generate the shared random projection used for nested p sweeps."""
    return rng.standard_normal((d, p_max))


# ----------------------------------------------------------------------
# 2. Fit + evaluate
# ----------------------------------------------------------------------

def fit_and_score(Xtr, ytr, Xte, yte, ridge=1e-6, max_iter=2000):
    """
    Fit logistic regression with a tiny L2 penalty (guarantees the
    estimate exists even when p >= n or data is separable -- pure
    unregularized MLE simply diverges there, see Candes & Sur 2020).
    Returns (train_logloss, test_logloss, train_err01, test_err01, converged).
    """
    # sklearn's C is inverse regularization strength: C = 1/(n*ridge)
    # roughly matches the ridge-penalized MLE with penalty `ridge`.
    C = 1.0 / (ridge * max(len(ytr), 1))
    clf = LogisticRegression(C=C, solver="lbfgs", max_iter=max_iter, tol=1e-6)

    # Guard against degenerate single-class training draws.
    if len(np.unique(ytr)) < 2:
        return np.nan, np.nan, np.nan, np.nan, False

    clf.fit(Xtr, ytr)
    converged = clf.n_iter_[0] < max_iter

    def logloss(X, y):
        p1 = clf.predict_proba(X)[:, 1]
        p1 = np.clip(p1, 1e-12, 1 - 1e-12)
        return -np.mean(y * np.log(p1) + (1 - y) * np.log(1 - p1))

    def err01(X, y):
        return np.mean(clf.predict(X) != y)

    return logloss(Xtr, ytr), logloss(Xte, yte), err01(Xtr, ytr), err01(Xte, yte), converged


# ----------------------------------------------------------------------
# 3. Monte Carlo grid with common random numbers
# ----------------------------------------------------------------------

def run_grid(
    X, y, n_list, p_list, B=50, test_frac=0.3, ridge=1e-6,
    feature_mode="random_features", base_seed=0,
):
    """
    Estimate g(n,p) and f(n,p) on a grid of (n,p), B Monte Carlo
    replicates each, with common random numbers shared across n and p
    within a replicate:

      - one fixed permutation per replicate -> nested training sets,
        so n and n+dn share the same first n points (CRN across n).
      - one fixed random feature map per replicate -> nested features,
        so p and p+dp share the same first p feature columns
        (CRN across p).

    Returns a dict of arrays shaped (len(n_list), len(p_list), B):
    'g_ll' (train logloss), 'f_ll' (test logloss),
    'g_01' (train 0/1 err), 'f_01' (test 0/1 err), 'converged'.
    """
    N, d = X.shape
    n_max = max(n_list)
    p_max = max(p_list)
    assert n_max + int(np.ceil(test_frac * n_max)) <= N or True, \
        "consider a larger held-out pool if n_max is close to N"

    shape = (len(n_list), len(p_list), B)
    out = {k: np.full(shape, np.nan) for k in
           ["g_ll", "f_ll", "g_01", "f_01"]}
    out["converged"] = np.zeros(shape, dtype=bool)

    n_test_fixed = int(np.ceil(test_frac * N))  # fixed-size held-out test pool

    for b in range(B):
        rng = np.random.default_rng(base_seed + b)
        perm = rng.permutation(N)
        test_idx = perm[:n_test_fixed]
        pool_idx = perm[n_test_fixed:]  # nested training pool, size N - n_test_fixed
        assert n_max <= len(pool_idx), "n_max too large given test_frac; shrink n_list or test_frac"

        W_full, _ = None, None
        if feature_mode == "random_features":
            W_full = make_W_full(d, p_max, rng)

        Xte_raw, yte = X[test_idx], y[test_idx]

        for i, n in enumerate(n_list):
            tr_idx = pool_idx[:n]  # nested: shared prefix across all n
            Xtr_raw, ytr = X[tr_idx], y[tr_idx]

            for j, p in enumerate(p_list):
                Xtr, _ = make_features(Xtr_raw, p, W_full, feature_mode, rng)
                Xte, _ = make_features(Xte_raw, p, W_full, feature_mode, rng)

                g_ll, f_ll, g_01, f_01, conv = fit_and_score(
                    Xtr, ytr, Xte, yte, ridge=ridge
                )
                out["g_ll"][i, j, b] = g_ll
                out["f_ll"][i, j, b] = f_ll
                out["g_01"][i, j, b] = g_01
                out["f_01"][i, j, b] = f_01
                out["converged"][i, j, b] = conv

    return out


# ----------------------------------------------------------------------
# 4. Finite-difference derivatives (paired, CRN-aware)
# ----------------------------------------------------------------------

def paired_central_diff(vals, axis_values, axis):
    """
    Central finite difference along `axis` (0=n, 1=p) of an array
    shaped (len(n_list), len(p_list), B), averaging the *per-replicate*
    difference before averaging over B (this is what lets CRN cancel
    noise -- averaging the raw values first and differencing second
    throws that benefit away).

    Returns (deriv_mean, deriv_se) each shaped like vals but with the
    swept axis shrunk by 2 (interior points only -> use one-sided
    diffs manually at the edges, e.g. near the p=n threshold).
    """
    axis_values = np.asarray(axis_values, dtype=float)
    vals = np.moveaxis(vals, axis, 0)  # sweep axis first
    diffs = []
    for k in range(1, len(axis_values) - 1):
        h_minus = axis_values[k] - axis_values[k - 1]
        h_plus = axis_values[k + 1] - axis_values[k]
        # non-uniform-grid-safe central difference
        d = (
            (vals[k + 1] - vals[k]) / h_plus * h_minus
            + (vals[k] - vals[k - 1]) / h_minus * h_plus
        ) / (h_minus + h_plus)
        diffs.append(d)
    diffs = np.stack(diffs, axis=0)  # (len-2, other_axis, B)
    deriv_mean = np.nanmean(diffs, axis=-1)
    deriv_se = np.nanstd(diffs, axis=-1) / np.sqrt(np.sum(~np.isnan(diffs), axis=-1))
    deriv_mean = np.moveaxis(deriv_mean, 0, axis if axis < deriv_mean.ndim else -1)
    deriv_se = np.moveaxis(deriv_se, 0, axis if axis < deriv_se.ndim else -1)
    return deriv_mean, deriv_se


def one_sided_diff(vals, axis_values, axis, side="left"):
    """One-sided difference at an edge (e.g. approaching p=n from below/above)."""
    axis_values = np.asarray(axis_values, dtype=float)
    vals = np.moveaxis(vals, axis, 0)
    if side == "left":
        h = axis_values[-1] - axis_values[-2]
        d = (vals[-1] - vals[-2]) / h
    else:
        h = axis_values[1] - axis_values[0]
        d = (vals[1] - vals[0]) / h
    d_mean = np.nanmean(d, axis=-1)
    d_se = np.nanstd(d, axis=-1) / np.sqrt(np.sum(~np.isnan(d), axis=-1))
    return d_mean, d_se


# ----------------------------------------------------------------------
# 5. Smoothed-surface derivatives (denoised alternative to raw finite diff)
# ----------------------------------------------------------------------

def fit_derivative_surface(n_list, p_list, mean_vals, mask=None, kx=3, ky=3, s=None):
    """
    Fit a smooth bivariate spline to the (n, p) -> mean error surface and
    return a callable giving analytic partial derivatives anywhere in
    the fitted region. Fit this SEPARATELY for p < n and p > n --
    logistic regression's error surface has a genuine non-smooth ridge
    at p = n, and smoothing across it will bias derivative estimates
    right where the effect is most interesting.

    mean_vals: 2D array (len(n_list), len(p_list)) of Monte-Carlo means
    (e.g. out["f_ll"].mean(axis=-1)).
    mask: optional boolean array same shape, e.g. (P < N) to restrict
    the fit to one side of the threshold.
    """
    N, P = np.meshgrid(n_list, p_list, indexing="ij")
    n_flat, p_flat, v_flat = N.ravel(), P.ravel(), mean_vals.ravel()
    if mask is not None:
        keep = mask.ravel() & np.isfinite(v_flat)
    else:
        keep = np.isfinite(v_flat)
    spline = SmoothBivariateSpline(
        n_flat[keep], p_flat[keep], v_flat[keep], kx=kx, ky=ky, s=s
    )

    def d_dn(n, p):
        return spline.ev(n, p, dx=1, dy=0)

    def d_dp(n, p):
        return spline.ev(n, p, dx=0, dy=1)

    return d_dn, d_dp


# ----------------------------------------------------------------------
# 6. Case builders -- the "bunch of different cases"
# ----------------------------------------------------------------------

def case_vary_n_fixed_p(n_list, p_fixed_list):
    """Case A: sweep n at several fixed p (classical data-scaling curves)."""
    return {p: n_list for p in p_fixed_list}


def case_vary_p_fixed_n(p_list, n_fixed_list):
    """Case B: sweep p at several fixed n, crossing p=n (double descent)."""
    return {n: p_list for n in n_fixed_list}


def case_threshold_zoom(center, halfwidth, num=15):
    """Case C: fine grid straddling the interpolation threshold p ~ n."""
    return np.unique(np.round(np.linspace(center - halfwidth, center + halfwidth, num)).astype(int))


def case_fixed_ratio_scaling(kappa, n_list):
    """
    Case D: move along a fixed ratio p/n = kappa (joint scaling). Useful
    for the total derivative d/dn at fixed complexity ratio, i.e. how
    error changes as you scale up n and p together, which is the
    regime the Sur-Candes / Deng-Kammoun-Thrampoulidis asymptotics
    describe directly (kappa = p/n held fixed as n,p -> infinity).
    """
    return [(n, max(1, int(round(kappa * n)))) for n in n_list]


# ----------------------------------------------------------------------
# Demo on synthetic data (sanity check the pipeline end-to-end)
# ----------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N, d_true = 4000, 40
    beta = rng.standard_normal(d_true) * 1.5
    X_raw = rng.standard_normal((N, d_true))
    logits = X_raw @ beta
    y = (rng.uniform(size=N) < 1 / (1 + np.exp(-logits))).astype(int)

    # Grid with multiple n's AND multiple p's, so we can take derivatives
    # along both axes. Runtime scales as len(n_list) * len(p_list) * B,
    # so keep B modest for a quick demo.
    n_list = [150, 200, 250, 300, 350, 400, 450]
    p_list = [10, 30, 60, 100, 150, 200, 250, 300, 350, 400, 500, 700]

    grid = run_grid(
        X_raw, y, n_list=n_list, p_list=p_list, B=25,
        feature_mode="random_features", base_seed=1,
    )

    f_mean = grid["f_ll"].mean(axis=-1)  # (len(n_list), len(p_list))
    g_mean = grid["g_ll"].mean(axis=-1)

    print("p values:", p_list)
    for i, n in enumerate(n_list):
        print(f"n={n:4d}  g(n,p)=", np.round(g_mean[i], 3))
    print()
    for i, n in enumerate(n_list):
        print(f"n={n:4d}  f(n,p)=", np.round(f_mean[i], 3))

    # --- derivatives w.r.t. p, at each interior n ---
    df_dp, df_dp_se = paired_central_diff(grid["f_ll"], p_list, axis=1)
    dg_dp, dg_dp_se = paired_central_diff(grid["g_ll"], p_list, axis=1)

    print("\ndf/dp (rows = n, cols = interior p points", p_list[1:-1], "):")
    for i, n in enumerate(n_list):
        print(f"  n={n:4d}:", np.round(df_dp[i], 5))

    print("\ndg/dp (rows = n, cols = interior p points", p_list[1:-1], "):")
    for i, n in enumerate(n_list):
        print(f"  n={n:4d}:", np.round(dg_dp[i], 5))

    # --- derivatives w.r.t. n, at each interior p ---
    df_dn, df_dn_se = paired_central_diff(grid["f_ll"], n_list, axis=0)
    dg_dn, dg_dn_se = paired_central_diff(grid["g_ll"], n_list, axis=0)

    print("\ndf/dn (rows = interior n", n_list[1:-1], ", cols = p):")
    for i, n in enumerate(n_list[1:-1]):
        print(f"  n={n:4d}:", np.round(df_dn[i], 5))

    print("\ndg/dn (rows = interior n", n_list[1:-1], ", cols = p):")
    for i, n in enumerate(n_list[1:-1]):
        print(f"  n={n:4d}:", np.round(dg_dn[i], 5))