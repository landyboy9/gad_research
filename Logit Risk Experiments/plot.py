import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from config import N_TRAIN, SEED, DATASET
from utils import features_to_param
 
 
train_loss_path = f"results/train_loss_{DATASET}_{N_TRAIN}_{SEED}.npy"
test_loss_path = f"results/test_loss_{DATASET}_{N_TRAIN}_{SEED}.npy"
n_vals_path = f"results/n_vals_{DATASET}_{N_TRAIN}_{SEED}.npy"
train_zero_one_path = f'results/train_zero_one_loss_{DATASET}_{N_TRAIN}_{SEED}.npy'
test_zero_one_path = f"results/test_zero_one_loss_{DATASET}_{N_TRAIN}_{SEED}.npy"

if not os.path.exists(test_loss_path) or not os.path.exists(train_loss_path) or not os.path.exists(n_vals_path):
    print(f"Couldn't find results files corresponding to configuration in 'config.py'. Run 'sweep.py' first to generate these files.")
    sys.exit(1)
 
train = [np.load(train_loss_path),np.load(train_zero_one_path)]
test = [np.load(test_loss_path),np.load(test_zero_one_path)]
n_vals = np.load(n_vals_path)
n_params = features_to_param(n_vals)
 
fig,ax=plt.subplots(2,1,figsize=(8,8))

labels=['Cross Entropy Loss','Zero-One Loss']

axes = [ax[0],ax[1]]
for i in range(2):
    ax[i].plot(n_params, test[i], marker='o', markersize=5, label='Test')
    ax[i].plot(n_params, train[i], marker='s',markersize=5, label='Train')
    ax[i].axvline(N_TRAIN, color='black', linestyle='--', label='Interpolation threshold')
    ax[i].set_xlabel("Number of parameters")
    ax[i].set_ylabel(labels[i])
    ax[i].legend()

plt.suptitle(f"{DATASET} | N_TRAIN={N_TRAIN} | SEED={SEED}")
plt.tight_layout()
 
os.makedirs("plots", exist_ok=True)
output_path = f"plots/loss_vs_features_{DATASET}_{N_TRAIN}_{SEED}.png"
plt.savefig(output_path, dpi=400)
print("Successfully saved plot")