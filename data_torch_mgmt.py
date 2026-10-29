# This file is for preparing the PyTorch tensors of the cleaned up data for
# training and also includes utilities for loading and batching the tensors.

import torch
import numpy as np
from torchdata.datapipes.iter import IterableWrapper

# File locations
f_data = "Data/"
f_torch = "Data/Torch/"
    
# Torch parameters
device = 'cpu'  # Running into VRAM limitations, just use CPU for this
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)

# INDICES
# 0. Settled turbidity
# 1. Influent turbidity
# 2. Influent conductivity
# 3. Coagulation basin average pH
# 4. Influent temperature
# 5. Alum dose
# 6. Flow rate in settling tank


# Function to create PyTorch tensor
def func_create_tensor(file_npy_cleaned, file_pt):
    
    # Load data and create a PyTorch tensor
    data_np = np.load(f_data + file_npy_cleaned)
    data_len, total_steps, total_dims = data_np.shape
    data_pt = torch.tensor(data_np, dtype=torch.float64, device=device)
    
    # Save full tensor
    torch.save(data_pt, f_torch + file_pt)
    print(f'Tensor saved to {f_torch}{file_pt}.\n')
  

# Function for 0-1 data normalization
def func_normalize_variables(f_torch, file_pt):
    print('Normalizing variables...')
    data_pt = torch.load(f_torch + file_pt)
    n_dims = data_pt.shape[-1]
    
    # Normalisation values are taken from an existing file made on the source
    #  plant from a previous project
    normvals = torch.load(f'{f_torch}/SP_1707-2007_normvals.pt')
    data_pt_n = torch.empty(data_pt.shape)
    for i in range(n_dims):
        minval, maxval = normvals[i,0], normvals[i,1]
        data_pt_n[:, :, i] = (data_pt[:, :, i] - minval)/(maxval - minval)

    # Save tensor as new file with "_n"(ormalized) appended to the file
    file_pt = file_pt.replace('.pt', '_n.pt')
    torch.save(data_pt_n, f_torch + file_pt)
    print(f'Tensor saved to {f_torch}{file_pt}.')
    
    
# Function to return normalized data to original
def func_unnormalize_variables(data, normvals):
    # Where data is an array of arbitrary size and normvals is a 2x1 vector
    # representing the min and max values of a given variable
    # Create empty tensor
    unnorm = torch.zeros(data.size())
    # Find min and max values in original data
    minval = normvals[0]
    maxval = normvals[1]
    # Transform back to original values
    unnorm = (data * (maxval - minval)) + minval          

    return unnorm
    

# Function to load tensors as a 4D array, where the first dimension of the array
# represents one of n blocks for time series validation
def func_loader(f_torch, file_pt, num_blocks=10+2):
    tensor_3d = torch.load(f_torch + file_pt).to('cpu')
    num_pts, total_steps, total_dims = tensor_3d.shape
    last_pt = num_pts - (num_pts % num_blocks) # <num_blocks pts are lost
    pts_per_block = int(last_pt / num_blocks)
    tensor_4d = tensor_3d[:last_pt, :, :].view(num_blocks, pts_per_block, total_steps, total_dims)
    return tensor_4d


# Function to split a tensor into minibatches
def func_batcher(batch_size, tensor_size):
    batcher = IterableWrapper(range(tensor_size))
    batcher = batcher.batch(batch_size=batch_size, drop_last=True)
    # Returns an iterable of minibatches for one epoch
    return iter(batcher)



#%% MAIN
if __name__ == '__main__':
    # Turn .npy array into .pt tensor
    file_npy_cleaned = "stn_2008-2307_cleaned.npy"
    file_pt = "stn_2008-2307_TRAIN.pt"
    func_create_tensor(file_npy_cleaned, file_pt)
    
    # Normalize values
    func_normalize_variables(f_torch, file_pt)