import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

# fetch dataset
df = np.load('bankruptcy.npy')
M = df[:, :-1]
y = df[:, -1]
num_obs, params = M.shape

rng=np.random.default_rng(51)

def resort_data(num_pts, num_params, M, y):
    '''
    num_pts: How many data points are needed (size of training data)
    M: Entire data matrix'''

    # Reorder M by observation so that logit doesn't break
    y_T = np.zeros(num_pts)
    while not np.any(y_T != y_T[0]):
        indices = rng.choice(np.arange(num_obs), size=num_pts, replace=False)
        y_T = y[indices]

    M_new = np.concatenate((M[indices, :], np.delete(M, indices, axis=0)))

    # Randomly sort parameters
    param_ordering = rng.permutation(params)
    M_new = M_new[:, param_ordering]

    # Breaking down the resorted matrix
    M_TM = M_new[:num_pts, :num_params]
    M_TU = M_new[:num_pts, num_params:]
    M_PM = M_new[num_pts:, :num_params]
    M_PU = M_new[num_pts:, num_params:]
    y_P = np.delete(y, indices)
    y_new = np.concatenate((y_T, y_P))
    return M_new, M_TM, M_TU, M_PM, M_PU, y_T, y_P, y_new

error_matrix = []

# Compute log loss (risk)
for j in range(2,num_obs,50):
    error_matrix.append([])
    for i in range(1,params,5):
        
        model=LogisticRegression(C=np.inf,solver='lbfgs')
        M_new, M_TM, M_TU, M_PM, M_PU, y_T, y_P, y_new = resort_data(j, i, M, y)
        model.fit(M_TM, y_T)
        # did_not_converge = any(issubclass(warn.category, ConvergenceWarning) for warn in w)
        # if not did_not_converge:
        y_hat = model.predict_proba(np.concatenate((M_TM,M_PM)))[:, 1]
        error = log_loss(y_new, y_hat, labels=[0,1])
        error_matrix[-1].append(error)
        # else:
        #     error_matrix[-1].append(np.nan)

error_matrix = np.array(error_matrix)

# Clip to 95th percentile
p95 = np.percentile(error_matrix, 95)
error_matrix = np.clip(error_matrix, a_min=0, a_max=p95)

# Plotting it
fig, ax = plt.subplots()
heatmap=ax.imshow(error_matrix, aspect='auto', origin='lower')
plt.xlabel("Number of Parameters")
plt.ylabel("Number of Data Points")
plt.title('Risk (Log Loss) Heatmap')
fig.colorbar(heatmap,label='Risk (Log Loss Function)')
plt.tight_layout()
plt.savefig(f'logit_risk_bankruptcy.png',dpi=400)