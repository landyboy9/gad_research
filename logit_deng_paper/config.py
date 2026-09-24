import numpy as np

SEED = 54
FIX_PLOT_SCALE = True # Set to True to fix scale of plot to that of the Deng paper


'''Specify the following variables according to the description in Deng paper'''
# Dimension of ambient space from which the data is sampled
d = 500
# Weird gamma that I dont understand
GAMMA = 2
# Ratio of d/n
ZETA = 3 
# Arbitrary fixed r values
R_VALUES = [2,5,10]
# Kappa values for each trial
KAPPA_VALUES = np.linspace(0, 3, 35)[1:]
# KAPPA_VALUES = np.unique(np.concatenate((np.geomspace(0.01, 0.5, 25),np.linspace(0.5, 3.0, 20))))
# Fixed number of parameters to use in the logistic model
p = 150
# How many "Monte Carlo" iterations to run
num_iter = 500


'''Configurations for sweep_heat.py'''
p_values = np.linspace(10,990,100,dtype=int)
n_values = np.linspace(11,990,100,dtype=int)

# n_values = [15000, 12743, 10827, 9198, 7815, 6639, 5640, 4792, 4071, 3459, 2938, 2496, 2121, 1802, 1531, 1300, 1105, 938, 797, 677, 575, 489, 415, 353, 300, 237, 196, 167, 146, 129, 116, 105, 96, 89, 82, 77, 72, 67, 64, 60, 57, 54, 52, 50]
# p_values = [45000, 38229, 32481, 27594, 23445, 19917, 16920, 14376, 12213, 10377, 8814, 7488, 6363, 5406, 4593, 3900, 3315,2000,1000,900,800,500,300,200,100,50,20,10]
r = 5


'''Configurations for sweep_trial.py'''
eta_norm = 1
big_sample_size = 100_000


# Set string for file name
FILE_STRING = f'deng_linear_model'

# Title for saved pyplot. For no title, set to None
# PLOT_TITLE = r'Regenerating $\mathbf{\beta}$ and $\mathbf{\gamma}$ each trial'
PLOT_TITLE=r"Deng et al.'s linear model (fig. 3)"
