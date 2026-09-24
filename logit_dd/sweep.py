from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, zero_one_loss
import load_data
from config import N_TRAIN, SEED, DATASET, ROOT, REP_COUNT
import numpy as np
import os
from utils import get_feature_counts, total_features, features_to_param, file_string

# Random generator with seed `SEED`
rng = np.random.default_rng(SEED)
total_num_features = total_features()
n_vals = get_feature_counts(total_num_features, n_left=15, n_right=15, power=2.2)

def train(X_train, X_test, y_train, num_features):
    '''Randomly select specified number of features to use in training set, and train the model.
       num_features: how many features to train with'''

    param_id = rng.choice(total_num_features, size=num_features, replace=False)

    X_train_sub = X_train[:, param_id]
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_sub, y_train)

    return model, X_train_sub, X_test[:, param_id]

# Initialize lists to store train and test losses for each repetition
train_loss = [[] for i in range(REP_COUNT)]
test_loss = [[] for i in range(REP_COUNT)]
train_zero_one_loss = [[] for i in range(REP_COUNT)]
test_zero_one_loss = [[] for i in range(REP_COUNT)]
AIC = [[] for i in range(REP_COUNT)]
BIC = [[] for i in range(REP_COUNT)]

data_gen_seeds = rng.integers(low=0, high=2**32 - 1, size=REP_COUNT)

# This is where the actual logistic regression happens
for k in range(REP_COUNT):
    X_train, y_train, X_test, y_test = load_data.load_mnist_subset(dataset_name=DATASET, n=N_TRAIN, seed=data_gen_seeds[k], root=ROOT)

    # Standardize training set
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0  # Prevents division by zero for constant border pixels
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    print(f'{k+1}/{REP_COUNT}')
    for i in range(len(n_vals)):
        model, X_train_sub, X_test_sub = train(X_train, X_test, y_train, n_vals[i])

        # Calculate cross entropy loss
        train_loss_i = log_loss(y_train, model.predict_proba(X_train_sub))
        test_loss_i = log_loss(y_test, model.predict_proba(X_test_sub))
        # Zero-one loss
        train_zero_one_loss_i = zero_one_loss(y_train, model.predict(X_train_sub))
        test_zero_one_loss_i = zero_one_loss(y_test, model.predict(X_test_sub))

        # Save the different losses
        train_loss[k].append(train_loss_i)
        test_loss[k].append(test_loss_i)
        train_zero_one_loss[k].append(train_zero_one_loss_i)
        test_zero_one_loss[k].append(test_zero_one_loss_i)

        neg_likelihood = log_loss(y_train, model.predict_proba(X_train_sub), normalize=False)

        # AIC calculation
        AIC_k = 2*features_to_param(n_vals[i]) + 2*neg_likelihood
        AIC[k].append(AIC_k)

        # BIC calculation
        BIC_k = features_to_param(n_vals[i])*np.log(N_TRAIN)+2*neg_likelihood
        BIC[k].append(BIC_k)

# Convert to numpy array
train_loss = np.array(train_loss)
test_loss = np.array(test_loss)
train_loss_mean = np.mean(train_loss, axis=0)
test_loss_mean = np.mean(test_loss, axis=0)
train_zero_one_loss = np.array(train_zero_one_loss)
test_zero_one_loss = np.array(test_zero_one_loss)
train_zero_one_loss_mean = np.mean(train_zero_one_loss, axis=0)
test_zero_one_loss_mean = np.mean(test_zero_one_loss, axis=0)
AIC = np.array(AIC)
BIC = np.array(BIC)

# Calculate Delta AIC within each subset draw (row k)
delta_AIC = AIC - np.min(AIC, axis=1, keepdims=True)
# Mean Delta AIC across all repetitions
delta_AIC_mean = np.mean(delta_AIC, axis=0)

# Calculate Delta BIC within each subset draw
delta_BIC = BIC - np.min(BIC, axis=1, keepdims=True)
delta_BIC_mean = np.mean(delta_BIC, axis=0)

# Save it all
os.makedirs("results", exist_ok=True)


np.save(f"results/train_loss{file_string()}.npy", train_loss_mean)
np.save(f"results/test_loss{file_string()}.npy", test_loss_mean)
np.save(f"results/n_vals{file_string()}.npy", n_vals)
np.save(f"results/train_zero_one_loss{file_string()}.npy", train_zero_one_loss_mean)
np.save(f"results/test_zero_one_loss{file_string()}.npy", test_zero_one_loss_mean)
np.save(f"results/delta_aic_mean{file_string()}.npy", delta_AIC_mean)
np.save(f'results/delta_bic_mean{file_string()}.npy', delta_BIC_mean)