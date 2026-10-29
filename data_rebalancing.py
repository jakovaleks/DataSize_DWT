import torch
import numpy as np
from scipy.interpolate import pchip_interpolate

# Consult the paper "Imbalanced regression and extreme value prediction" by
# Ribeiro and Moniz (2020) for a detailed description of the relevance func

# Control points - y_k [influent turbidity, in units of NTU]
y_k = np.array([0, 5.7, 15, 30, 100])
# Control point relevance values - phi(y_k)
phi_y_k = np.array([0.01, 0.01, 0.2, 1, 1])

# Light rebalancing
# y_k = np.array([0, 5.7, 10, 100])
# phi_y_k = np.array([0.2, 0.2, 1, 1])

# Moderate rebalancing
# y_k = np.array([0, 5.7, 10, 100])
# phi_y_k = np.array([0.05, 0.05, 1, 1])

# Heavy rebalancing
# y_k = np.array([0, 5.7, 15, 30, 100])
# phi_y_k = np.array([0.01, 0.01, 0.2, 1, 1])

# Get median influent turbidity value from each training example
trn_data = torch.load('Data/Torch/stn_2008-2307-TRAIN_n.pt')
medians = torch.median(trn_data[:, :, 1], axis=1)[0].numpy()

# Define a relevance function from the above control points
relevance_func = pchip_interpolate(y_k, phi_y_k, medians, der=0)

# For each training example, roll a random number from 0 to 1
# If the random number is greater than the value of the relevance function
#  for that training example, remove it from the training data
random_nums = np.random.rand(len(medians))
flags = torch.tensor(random_nums < relevance_func)
trn_data = trn_data[flags, :, :]
torch.save(trn_data, 'Data/Torch/stn_2008-2307-TRAIN_n_rebalanced_heavy.pt')
