# Set DATASET to be "mnist", "cifar10", or "svhn"
DATASET = "mnist"
# Specify the root directory for the dataset
ROOT = "C:/Users/lando/Projects/GAD Research/catdd/mnist_dd/notebooks/mnist_data"

'''
Landon's laptop directories:
    mnist - C:/Users/lando/Projects/GAD Research/catdd/mnist_dd/notebooks/mnist_data
    cifar10 - C:/Users/lando/Projects/GAD Research/catdd/logit_dd/data
    svhn - C:/Users/lando/Projects/GAD Research/catdd/logit_dd/data
'''

# Specify the number of repetitions to run for each configuration (for averaging results)
REP_COUNT = 5
# N_TRAIN = number of training points (Belkin = 4000)
N_TRAIN = 1000
# random seed for choosing training points
SEED = 42
# Specify whether to 'zoom in' on the range of feature counts to be tested (True) or to test the full range (False)
ZOOM = True