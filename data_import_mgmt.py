# This file is for converting .csv files into one .npy file.

import pandas as pd
import numpy as np
import os, fnmatch

# File locations
f_data = "Data/"
f_csv = f"{f_data}Subset/"

# The data are stored as .csv files from each month;
# For example, stn_2008.csv is the data for August 2020.
# The first column is a string representing the date and time as YYYY-MM-DD HH:MM:SS (Montreal time).
# The remaining columns are the variables measured at that timestamp
# (e.g. influent temperature, alum dose, effluent turbidity, etc.).
list_csv = sorted(fnmatch.filter(os.listdir(f_csv), "stn_*.csv"))
start_date = list_csv[0][list_csv[0].find('_')+1:list_csv[0].find('.')]
end_date = list_csv[-1][list_csv[-1].find('_')+1:list_csv[-1].find('.')]
# Overall dataset ranges from August 2020 to July 2024
# Files are made separately depending on need by putting the necessary
#  monthly .csv files in Data/Subset/
# At minimum in this case, one file is Aug '20 - Jul '23 (training)
#  and another is Aug '23 - Jul '24 (testing)
file_npy = f"{f_data}stn_{start_date}-{end_date}.npy"
# Append "_sorted" for the stacked and sorted npy file
file_npy_sorted = file_npy[:file_npy.rfind('.')] + '_sorted' + file_npy[file_npy.rfind('.'):]

# Total time steps per example (for training)
n_steps = 45
crit_steps = 35

# Function to convert all the individual .csv files into one .npy file
def func_csv_to_npy(f_csv, list_csv, file_npy):
    print('Copying csv files to npy file...')
    array_pd = pd.DataFrame()
    # Iterate through all the files in list_csv
    for date_file in list_csv:
        data_pd = pd.read_csv(f_csv + date_file, index_col=0, parse_dates=True, 
                              encoding = 'unicode_escape', engine ='python')
        # Create a pandas dataframe of all the data
        if date_file == list_csv[0]:
            array_pd = data_pd
        else:
            array_pd = pd.concat((array_pd, data_pd))
        print(f'Finished copying data for {date_file}.')
    
    # Create a sorted version (akin to what the torch tensor needs to be)
    
    # Add timezone info, convert timestamps to EST, and remove timezone info
    # to avoid problems caused by original timestamps being in DST
    # (this means that the new summer timestamps will be 1 hour behind)
    print('Fixing timezone data...')
    array_pd = array_pd.tz_localize('America/Montreal', ambiguous='infer')
    array_pd = array_pd.tz_convert('EST')
    array_pd = array_pd.tz_localize(None)
    
    # Convert pandas dataframe to numpy array and save
    array_np_idx = array_pd.index.to_numpy(dtype='object')
    array_np_idx = np.expand_dims(array_np_idx, -1)
    array_np = array_pd.to_numpy()
    array_np = np.concatenate((array_np_idx, array_np), -1)
    np.save(file_npy, array_np, allow_pickle=True)
    print(f'Saved to {file_npy}.\n')
    
    # Create separate array for each settler
    # This step removes the timestamps
    print('Creating new arrays...')
    n_settlers = 3
    array1 = array_np[:, [1, 2, 3, 4, 5, 6, 7, 8]]
    array2 = array_np[:, [1, 2, 3, 4, 5, 9, 10, 11]]
    array3 = array_np[:, [1, 2, 3, 4, 5, 12, 13, 14]]
    
    # Reshape array to (training_examples, n_settlers [3], n_steps [45], n_features [7])
    array_exp_shape = ((array1.shape[0] - n_steps, n_settlers, n_steps, array1.shape[1]))
    arrays_exp = np.zeros(array_exp_shape)
    flag_array = np.ones((array1.shape[0] - n_steps, n_settlers), dtype=bool)
    
    initial_array_list = [array1, array2, array3]
    for i, arr in enumerate(initial_array_list):
    # Add extra dimension to create examples of multiple time steps
    # New array size is (n_examples, n_settlers, n_steps, n_dims)
        for j in range(array_exp_shape[0]):
            for k in range(n_steps):
                arrays_exp[j, i, k, :] = arr[j+k, :]
            # Flag any instances where the settler state is 0 (off) at any
            # point during the last crit_steps time steps (in this case 35)
            if sum((arrays_exp[j, i, :, -1])[-crit_steps:] == 0) > 0:
                flag_array[j, i] = 0
        print(f'Finished settler {i+1}.')
    
    # Merge along the first dimension (already sorted by timestamp because
    # of how shape was chosen)
    arrays_exp = np.reshape(arrays_exp, (n_settlers * (array1.shape[0] - n_steps), n_steps, -1))
    flag_array = np.reshape(flag_array, -1)
    
    # Arrange order of variables to match tensors from previous project
    # (0. Settled turbidity, 1. Influent turbidity, 2. Influent conductivity,
    #  3. Coagulation pH, 4. Influent temperature, 5. Alum dose, 6. Flow rate)
    # and remove the settler state variable
    # Also removed the flagged examples
    print('Rearranging...')
    arrays_exp = arrays_exp[flag_array, :, :]
    new_arrays_exp = arrays_exp[:, :, [5, 0, 2, 3, 1, 4, 6]]
    
    # Save npy file
    np.save(file_npy_sorted, new_arrays_exp)
    print(f'Saved to {file_npy_sorted}.\n')


#%% MAIN

if __name__ == '__main__':
    list_csv = sorted(fnmatch.filter(os.listdir(f_csv), "stn_*.csv"))
    print(f"Ranges from {list_csv[0]} to {list_csv[-1]}")
    # Data ranges from August 2020 (2008) to July 2024 (2407)
    file_npy = f"{f_data}stn_2008-2307.npy"
    # Append "_sorted" for the stacked and sorted npy file
    file_npy_sorted = file_npy[:file_npy.rfind('.')] + '_sorted' + file_npy[file_npy.rfind('.'):]
    func_csv_to_npy(f_csv, list_csv, file_npy)