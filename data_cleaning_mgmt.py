# This file is for cleaning up the data based on various heuristics that have
#  been deemed appropriate for better model training.
# Data where zero or a negative value have been recorded are removed.
# Data where the delta between two consecutive time steps exceed an
#  acceptable threshold for any sensor variable are removed.
# Data where values fall outside normal limits/LoD for those instruments are
#  clamped as follows: settled turbidity <= 10, influent turbidity <= 100,
#  4 <= coagulation pH <= 8, flow rate <= 100.

import numpy as np

# File locations
f_data = "Data/"

# ============================ #
# Indices ==================== #
# 0. Settled turbidity ======= #
# 1. Influent turbidity ====== #
# 2. Influent conductivity === #
# 3. Coagulation pH ========== #
# 4. Influent temperature ==== #
# 5. Alum dose =============== #
# 6. Flow rate =============== #
# ============================ #  


# Function to perform the data cleanup
def func_clean_data(file_npy, file_npy_cleaned):
    # Load data
    array_np = np.load(file_npy)
    
    # Flag all examples (0 index) that contain any zero/negative values
    flags1 = np.all((array_np > 0), (1, 2))
    
    # Flag all examples with sudden spikes in any of the first five variables
    # (this suggests some kind of sensor error)
    # (the acceptable thresholds between consecutive time steps
    #  were determined via visual examination of the data - see above indices)
    delta_thresholds = np.array([0.4, 3, 2.5, 0.5, 0.035])
    deltas = np.abs(array_np[:, 1:, :5] - array_np[:, :-1, :5])
    max_deltas = np.max(deltas, axis=1)
    flags2 = np.all(max_deltas < delta_thresholds, axis=1)
    
    # Data from this plant were only logged to 3 decimal places rather than 10,
    # although looking at the data there don't seem to be any instances of
    # communication error resulting in frozen sensor readings
    
    # Remove flagged examples from the array
    # (i.e. keep only examples that don't contain any zero values and
    #  whose deltas don't exceed the acceptable threshold)
    flags = flags1 * flags2
    
    array_np = array_np[flags, :, :]
    
    # Clamp the variables that require it (based on detectable range)
    # Settled turbidity <= 10
    array_np[:, :, 0] = np.clip(array_np[:, :, 0], a_min=None, a_max=10)
    # Influent turbidity <= 100
    array_np[:, :, 1] = np.clip(array_np[:, :, 1], a_min=None, a_max=100)
    # 4 <= Coagulation pH <= 8
    array_np[:, :, 3] = np.clip(array_np[:, :, 3], a_min=4, a_max=8)
    # Flow rate <= 100
    array_np[:, :, 6] = np.clip(array_np[:, :, 6], a_min=None, a_max=100)
    
    # Save data to a new file
    np.save(file_npy_cleaned, array_np)
    

#%% MAIN

if __name__ == '__main__':
    # Append "_cleaned" for new file name after cleaning
    file_npy = f_data + "stn_2008-2307_sorted.npy"
    file_npy_cleaned = file_npy[:file_npy.rfind('_sorted.npy')] + '_cleaned.npy'
    func_clean_data(file_npy, file_npy_cleaned)