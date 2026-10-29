# This file is for testing the final models.

import torch
import os
import numpy as np
from data_torch_mgmt import func_batcher, func_unnormalize_variables
from torch import nn
from natsort import natsorted

# File locations
f_data = "Data/Torch/"
f_model = "Models/2test/" # Location containing all the trained models
    
# Torch parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)


# Class to test the model
class model_tester:
    def __init__(self, f_data, f_model):
        
        # Specify folders locations
        self.f_data = f_data
        self.f_model = f_model
        
        
    def initialize(self, 
                   production_data_filename,
                   master_values_filename, 
                   batch_size=64
                   ):
        
        # Define tensor and batch sizes
        self.batch_size = batch_size
        self.output_steps = 15
        
        # Load production data set
        prod_data = torch.load(self.f_data + production_data_filename)
        self.production_data_filename = production_data_filename
        self.examples, self.total_steps, self.total_dims = prod_data.shape
        self.input_steps = self.total_steps - self.output_steps
        self.data_in = prod_data[:, :self.input_steps, :].to(device)
        self.true_out = prod_data[:, self.input_steps:, 0].to(device)
        self.master_vals = torch.load(f_data + master_values_filename)
        print(f'Evaluating models on {f_data}{production_data_filename}')
        
    def test(self):

        # Load all the models in the f_model directory
        model_names = [i for i in os.listdir(f_model) if '.pt' in i]
        model_names = natsorted(model_names)
        
        # Iterate through the models and test each one
        for model_name in model_names:
            mdl = torch.load(f_model + model_name).to(device)
            loss_tst, residuals_tst = [], []
            batch_gen = func_batcher(self.batch_size, self.examples)

            with torch.no_grad():
                # Iterate through all the batches in this epoch
                for bat in batch_gen:
                    x_bat, y_bat = self.data_in[bat, :, :], self.true_out[bat, :]
                    mdl.teacher_forcing_ratio = 0
                    preds_tst = (mdl.forward(x_bat, y_bat)).to(device)
                    rmse_tst, mae_tst = self.func_loss(preds_tst, y_bat.view((self.batch_size, self.output_steps, 1)), self.master_vals[0, :])
                    loss_tst.append(rmse_tst.tolist())
                    residuals_tst.append(mae_tst.tolist())
                loss_tst.append(rmse_tst.tolist())
                residuals_tst.append(mae_tst.tolist())
                
            final_rmse = np.mean(loss_tst)
            final_mae = np.mean(residuals_tst)
            
            prod_summary_file = "_production_summary.txt"
            if os.path.exists(prod_summary_file) == False:
                with open(prod_summary_file, "x") as f:
                    f.write("Model" + " "*38 + "Month"+ " "*23 + "RMSE     MAE\n")
            with open(prod_summary_file, "a") as f:
                f.write(f"{model_name:<43}{self.production_data_filename}    {final_rmse:.4f}   {final_mae:.4f}\n")
            
            print(f'Evaluted {model_name}: production set RMSE = {final_rmse:.4f} NTU, MAE = {final_mae:.4f} NTU.')

    # Function to calculate loss in NTU
    def func_loss(self, preds, actual, master, testing=True):
        loss = nn.MSELoss()
        mae = None
        # If testing we also want MAE loss, otherwise only MSE
        if testing: loss_MAE = nn.L1Loss(reduction='none')
        # Get un-normalized values
        preds_real = func_unnormalize_variables(preds, master)
        actual_real = func_unnormalize_variables(actual, master)
        # Compute RMSE in original units (real space - NTU in the case of turbidity))
        rmse = torch.sqrt(loss(preds_real, actual_real)).to(device)
        if testing: mae = loss_MAE(preds_real, actual_real)
        return rmse, mae

#%% MAIN

if __name__ == '__main__':
    # Set production data as needed based on what is being evaluated
    # mfiles = list(['0124', '0224', '0324', '0424', '0524', '0624', '0724', '0823', '0923', '1023', '1123', '1223'])
    # mfiles = list(['10-NTU', '15-NTU', '20-NTU', '25-NTU'])
    # for m in mfiles:
        tester = model_tester(f_data, f_model)
        tester.initialize(
                          production_data_filename="stn_2308-2407_PROD_n.pt",
                          # production_data_filename=f"stn_{m}_PROD_n.pt",
                          master_values_filename="SP_1707-2007_normvals.pt",
                          batch_size=64
                          )
        tester.test()
    