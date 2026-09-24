import numpy as np
import config


def features_to_param(x):
    return 10*(x+1)

def param_to_features(x):
    return (x/10)-1

def get_feature_counts(high, n_left, n_right, power=2.2):
    """
    Return a sorted list of feature counts between 1 and `high`,
    denser near `threshold` and sparser toward the edges.
    """
    if config.ZOOM:
        listy = param_to_features(np.linspace(config.N_TRAIN-500, config.N_TRAIN+500, num=30)).astype(int)
        if not np.any(listy == config.N_TRAIN):
            idx = np.searchsorted(listy, int(param_to_features(config.N_TRAIN)))
            listy = np.insert(listy, idx, int(param_to_features(config.N_TRAIN)))
        return listy

    threshold = int(config.N_TRAIN/10)-1

    u_left = np.linspace(0, 1, n_left)
    dist_left = (threshold - 1) * u_left**power
    left = np.round(threshold - dist_left).astype(int)

    u_right = np.linspace(0, 1, n_right)
    dist_right = (high - threshold) * u_right**power
    right = np.round(threshold + dist_right).astype(int)

    features = sorted(set(left.tolist()) | set(right.tolist()) | {threshold})
    return [f for f in features if 1 <= f <= high]

def total_features():
    return 784 if config.DATASET == "mnist" else 1024

def file_string():
    if config.ZOOM:
        return f'_{config.DATASET}_{config.N_TRAIN}_{config.SEED}_zoomed'
    return f'_{config.DATASET}_{config.N_TRAIN}_{config.SEED}'