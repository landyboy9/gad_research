import numpy as np
import config

# Given dataset, calculate the 
def calc_n_vals(X):
    num_obs, num_params = X.shape





def get_feature_counts(X_train, high, n_left, n_right, power=2.2):
    """
    Return a sorted list of feature counts between 1 and `high`,
    denser near `threshold` and sparser toward the edges.
    """
    threshold, high = X_train.shape

    u_left = np.linspace(0, 1, n_left)
    dist_left = (threshold - 1) * u_left**power
    left = np.round(threshold - dist_left).astype(int)

    u_right = np.linspace(0, 1, n_right)
    dist_right = (high - threshold) * u_right**power
    right = np.round(threshold + dist_right).astype(int)

    features = sorted(set(left.tolist()) | set(right.tolist()) | {threshold})
    return [f for f in features if 1 <= f <= high]