import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from config import N_TRAIN, SEED, DATASET
from utils import features_to_param, file_string
 
 
train_loss_path = f"results/train_loss{file_string()}.npy"
test_loss_path = f"results/test_loss{file_string()}.npy"
n_vals_path = f"results/n_vals{file_string()}.npy"
train_zero_one_path = f'results/train_zero_one_loss{file_string()}.npy'
test_zero_one_path = f"results/test_zero_one_loss{file_string()}.npy"
aic_path = f'results/delta_aic_mean{file_string()}.npy'
bic_path = f'results/delta_bic_mean{file_string()}.npy'

if not os.path.exists(test_loss_path) or not os.path.exists(train_loss_path) or not os.path.exists(n_vals_path) or not os.path.exists(aic_path) or not os.path.exists(bic_path):
    print(f"Couldn't find results files corresponding to configuration in 'config.py'. Run 'sweep.py' first to generate these files.")
    sys.exit(1)
 
train = [np.load(train_loss_path),np.load(train_zero_one_path)]
test = [np.load(test_loss_path),np.load(test_zero_one_path)]
n_vals = np.load(n_vals_path)
n_params = features_to_param(n_vals)
aic = np.load(aic_path)
bic = np.load(bic_path)
 
fig,ax=plt.subplots(2,1,figsize=(8,8))
labels=['Cross Entropy Loss','Zero-One Loss']

axes = [ax[0],ax[1]]
for i in range(2):
    ax[i].plot(n_params, test[i], marker='o', markersize=5, label='Test')
    ax[i].plot(n_params, train[i], marker='s',markersize=5, label='Train')
    ax[i].axvline(N_TRAIN, color='black', linestyle='--', label='Interpolation threshold')
    ax[i].axvline(n_params[np.argmin(aic)], color='forestgreen', linestyle='--', label=r'Min $\Delta\text{AIC}$ value')
    ax[i].axvline(n_params[np.argmin(bic)], color='darkviolet', linestyle='--', label=r'Min $\Delta\text{BIC}$ value')
    ax[i].set_xlabel("Number of parameters")
    ax[i].set_ylabel(labels[i])
    if i==0:
        ax[i].legend()

plt.suptitle(f"{DATASET} | N_TRAIN={N_TRAIN} | SEED={SEED}")
plt.tight_layout()
 
os.makedirs("plots", exist_ok=True)
output_path = f"plots/loss_vs_features{file_string()}.png"
plt.savefig(output_path, dpi=400)
print("Successfully saved plot")