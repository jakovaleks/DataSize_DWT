# This file is for training RNN models.

import torch
import models
import time
import numpy as np
import matplotlib.pyplot as plt
from torch import optim, nn
from data_torch_mgmt import func_loader, func_batcher

# File locations
f_root = "/home/aj/Documents/Research/Transfer/"
f_data = "Data/Torch/"
f_model = "Models/"
f_model_orig = "Models/SP_4retraining.pt"   # Location of model from previous project
fileloc_data = f"{f_root}{f_data}"
fileloc_model = f"{f_root}{f_model}"
    
# Torch parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)


# Class to run the training procedure
class rnn_trainer:
    def __init__(self, fileloc_data, fileloc_model, model_prefix="MDL_", existing_model=None, interm=None):
        
        # Specify folders to load/save data/models to/from
        self.fileloc_data = fileloc_data
        self.fileloc_model = fileloc_model
        self.existing_model = existing_model
    
        self.model_prefix = model_prefix
        self.interm = interm
    
    # Load the training data and initialize the model
    def initialize(self,
                   training_data_filename,
                   batch_size=1024,
                   num_training_blocks=10,
                   hidden_dims=64,
                   num_layers=1,
                   frozen_layers=2
                   ):
        
        # These are hard-coded to match the previous project
        self.total_dims = 7
        self.input_steps = 30
        self.output_steps = 15
        self.total_steps = self.input_steps + self.output_steps
        self.encoder_dropout_ratio = 0
        self.decoder_dropout_ratio = 0
        self.teacher_forcing_ratio = 0.5
        
        self.num_training_blocks = num_training_blocks
        
        # If provided, load a previously-trained model
        if self.existing_model is not None:
            self.from_scratch = False
            self.mdl = torch.load(f"{f_root}{f_model}{self.existing_model}")
            self.training_data_filename = training_data_filename
            
            # The following are ignored from the function input and instead
            # taken directly from the existing model
            self.batch_size = self.mdl.batch_size
            self.hidden_dims = self.mdl.hidden_dims
            self.num_layers = self.mdl.num_layers
            
            # Freeze layers if specified
            self.frozen_layers = frozen_layers
            # Based on the torch GRU implementation, 1 layer = 2 weights + biases (4 total)
            if self.frozen_layers > 0:
                frozen_params = self.frozen_layers * 4 # See comment above
                frozen_counter = 0
                for p in self.mdl.encoder.gru.parameters():
                    frozen_counter += 1
                    # Freeze the first n layers (in this case 2)
                    if frozen_counter <= frozen_params:
                        p.requires_grad = False
                    # Optionally re-initialize the remaining layer(s) (not used in this case)
                    # else:
                    #     nn.init.uniform_(p, a=-1/np.sqrt(self.hidden_dims), b=1/np.sqrt(self.hidden_dims))
                frozen_counter = 0
                for p in self.mdl.decoder.gru.parameters():
                    frozen_counter += 1
                    # Freeze the first n layers (in this case 2)
                    if frozen_counter <= frozen_params:
                        p.requires_grad = False
                    # Optionally re-initialize the remaining layer(s) (not used in this case)
                    # else:
                    #     nn.init.uniform_(p, a=-1/np.sqrt(self.hidden_dims), b=1/np.sqrt(self.hidden_dims))
                # Optionally re-initialize the output layer (not used in this case)    
                # for p in self.mdl.decoder.affine.parameters():
                #     nn.init.uniform_(p, a=-1/np.sqrt(self.hidden_dims), b=1/np.sqrt(self.hidden_dims))
        
        # Otherwise define a new model    
        else:
            self.from_scratch = True
            self.training_data_filename = training_data_filename
            self.batch_size = batch_size
            self.hidden_dims = hidden_dims
            self.num_layers = num_layers
            
        # Concatenate into a tuple
        self.mdl_parameters = (self.total_dims,
                               self.hidden_dims,
                               self.num_layers,
                               self.batch_size,
                               self.input_steps,
                               self.output_steps,
                               self.encoder_dropout_ratio,
                               self.decoder_dropout_ratio,
                               self.teacher_forcing_ratio,
                               )
        self.mdl_parameter_names = ['Total dimensions',
                                    'Hidden dimensions',
                                    'Hidden layers',
                                    'Batch size',
                                    'Input time steps',
                                    'Output time steps',
                                    'Encoder dropout ratio',
                                    'Decoder dropout ratio',
                                    'Teacher forcing ratio',
                                    ]
            
        # Load a new model if no existing one was provided
        if self.existing_model is None:
            self.mdl = models.GRU(*self.mdl_parameters).to(device)
            
        # Initialize placeholder for best model
        self.best_mdl = models.GRU(*self.mdl_parameters).to('cpu')
        self.best_mdl.load_state_dict(self.mdl.state_dict())
        
        # Load the data as a 4D array
        self.xy = func_loader(self.fileloc_data, self.training_data_filename, self.num_training_blocks + 2)
        self.points_per_block = self.xy.shape[1]
        
    # Function to progressively unfreeze frozen layers  
    def func_thaw(self, layers_to_keep_frozen):
        # Based on the torch GRU implementation, 1 layer = 2 weights + biases (4 total)
        frozen_params = layers_to_keep_frozen * 4
        frozen_counter = 0
        for p in self.mdl.encoder.gru.parameters():
            frozen_counter += 1
            # Skip over the first n layers and unfreeze later ones
            if frozen_counter > frozen_params:
                p.requires_grad = True
        frozen_counter = 0
        for p in self.mdl.decoder.gru.parameters():
            frozen_counter += 1
            # Skip over the first n layers and unfreeze later ones
            if frozen_counter > frozen_params:
                p.requires_grad = True
        self.opt = optim.Adam(filter(lambda p: p.requires_grad, self.mdl.parameters()), lr=self.learning_rate, weight_decay=self.weight_decay)

    # Train the model
    def train(self,
              learning_rate=5e-4,
              max_iters_per_block=250,
              max_iters_increasing_validation_loss=10,
              weight_decay=1e-8,
              gradient_clipping_norm=1,
              blocked_crossval=True
              ):
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Load optimizer
        if self.from_scratch == True:
            self.opt = optim.Adam(self.mdl.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            self.opt = optim.Adam(filter(lambda p: p.requires_grad, self.mdl.parameters()), lr=learning_rate, weight_decay=weight_decay)
        self.max_iters_per_block = max_iters_per_block
        self.blocked_crossval = blocked_crossval
        print(f'Training {self.mdl.desc}\nOptimization to be performed using {type(self.opt)}.\n')
        
        # Initialize loss logs for trn and val as empty lists
        self.loss_trn, self.loss_val = [], []
        
        # Initialize loss log for tst as a 2 column array [final_block_iter, test_set_rmse]
        self.loss_tst = np.zeros((self.num_training_blocks, 2))
        self.total_iters = 0
        
        # Train using rolling-origin cross-validation
        start_time = time.perf_counter()
        self.stuck_in_zero_grad = False
        
        # If blocked_crossval == True:
        # Loop through n-2 rolling blocks (on the last iteration, the valdiation
        # and test blocks are at the end of the data set)
        # If blocked_crossval == False:
        # Immediately assign the first n-2 blocks as the training set (i.e. equivalent
        # to a standard cross-validation)
        if self.blocked_crossval == True:
            training_range = range(self.num_training_blocks)
        else:
            training_range = range(self.num_training_blocks - 1, self.num_training_blocks)
            
        for block in training_range:
            # If freezing, thaw layers after the third and seventh block
            # The choice is arbitrary and based on the total number of
            #  training blocks (10 in this case) as well as the total number
            #  of layers in the model (3 in this case) - you can add more
            #  elif statements for more layers if desired
            if self.frozen_layers > 0:
                if block == 3:
                    self.func_thaw(1)
                    print("Thawing second layer...")
                elif block == 7:
                    self.func_thaw(0)
                    print("Thawing first layer...")
            # Load training set for the current block
            # These are loaded to the CPU to avoid VRAM issues
            xy_trn = self.xy[:block + 1, :, :, :].view(((block + 1) * self.points_per_block,
                                                        self.total_steps,
                                                        self.total_dims)
                                                        ).to('cpu')
            x_trn = (xy_trn[:, :-self.output_steps, :]).to('cpu')
            y_trn = (xy_trn[:, -self.output_steps:, :]).to('cpu')
            
            # Load validation data for the current block
            xy_val = self.xy[block + 1, :, :, :].view((self.points_per_block,
                                                        self.total_steps,
                                                        self.total_dims)
                                                        ).to('cpu')
            # Remove the first total_steps examples to avoid data leakage
            x_val = (xy_val[self.total_steps:, :-self.output_steps, :]).to(device)
            # y_val only needs the turbidity as teacher forcing is not used
            y_val = (xy_val[self.total_steps:, -self.output_steps:, 0]).to(device)
            
            # Shuffle the order of examples in x_trn and y_trn
            trn_idx = torch.randperm((block + 1) * self.points_per_block)
            x_trn = x_trn[trn_idx].to(device)
            y_trn = y_trn[trn_idx].to(device)
                
            # Iterate training loop until either max_iters_per_block is reached,
            # or validation loss is above the lowest loss for that block for
            # max_iters_increasing_validation_loss iterations
            for i in range(max_iters_per_block):
                self.mdl.train()
                
                # Set the teacher forcing ratio for training
                self.mdl.teacher_forcing_ratio = self.teacher_forcing_ratio
                
                # Create a "batch generator" that will load all the batches for
                # this iteration (epoch)
                batch_gen = func_batcher(self.batch_size,
                                         (block + 1) * self.points_per_block
                                         )
                
                # Empty the iteration loss list for this training iteration
                loss_iter = []
                
                # Iterate through all the training batches in this epoch
                for bat in batch_gen:
                    self.opt.zero_grad()
                    
                    # Load the input/output data in the batch
                    x_bat, y_bat = x_trn[bat, :, :], y_trn[bat, :, :]
                    
                    # Perform forward pass on the data
                    preds_trn = (self.mdl.forward(x_bat, y_bat)).to(device)
                    
                    # Calculate training loss on this batch
                    # We transform it back to NTU to always be comparable!
                    rmse_trn = self.func_loss(preds_trn, y_bat[:, :, 0].view((self.batch_size, self.output_steps, 1)))
                                        
                    # Save the training loss to the list
                    loss_iter.append(rmse_trn.tolist())
                    
                    # Perform backwards pass
                    rmse_trn.backward()
                    torch.nn.utils.clip_grad_norm_(self.mdl.parameters(), gradient_clipping_norm)
                    self.opt.step()
                    
                # Save the average loss for this iteration to the training loss list
                self.loss_trn.append(np.mean(loss_iter))
                
                # Evaluate model performance on the validation set
                self.mdl.eval()
                
                # Disable teacher forcing for validation
                self.mdl.teacher_forcing_ratio = 0
                
                batch_gen = func_batcher(self.batch_size,
                                         self.points_per_block - self.total_steps
                                         )
                loss_iter = []
                
                with torch.no_grad():
                    # Iterate through all the validation batches in this epoch
                    for bat in batch_gen:
                        x_bat, y_bat = x_val[bat, :, :], y_val[bat, :]
                        preds_val = (self.mdl.forward(x_bat, y_bat)).to(device)
                        rmse_val = self.func_loss(preds_val, y_bat.view((self.batch_size, self.output_steps, 1)))
                        loss_iter.append(rmse_val.tolist())
                        
                    self.loss_val.append(np.mean(loss_iter))
                
                # Increment total_iters by 1 (used for performance plotting)
                self.total_iters += 1
                
                # Find the minimum validation loss in this block
                block_min_val = np.min(self.loss_val[-(i + 1):])
                
                # If the last iteration had the minimum validation loss,
                # save it as self.best_mdl
                if block_min_val == self.loss_val[-1]:
                    self.best_mdl.load_state_dict(self.mdl.state_dict())
                    print(f'Training block {block + 1}/{self.num_training_blocks}, iter {i + 1}:\t Training loss: {self.loss_trn[-1]:.4f}\t Validation loss: {self.loss_val[-1]:.4f} <---')
                else :
                    print(f'Training block {block + 1}/{self.num_training_blocks}, iter {i + 1}:\t Training loss: {self.loss_trn[-1]:.4f}\t Validation loss: {self.loss_val[-1]:.4f}')
                    
                # Check to see if max_iters_increasing_validation_loss has
                # been exceeded (i.e. the minimum validation loss in the block
                # must have occured in the last that many iterations)
                if i >= max_iters_increasing_validation_loss:
                    recent_vals = np.array(self.loss_val[-max_iters_increasing_validation_loss:])
                    
                    # If the limit has been exceeded, exit the training loop
                    # for this block
                    if (recent_vals > block_min_val).all():
                        break
                    
                    # It's possible that the model was initialized to an area
                    # of zero gradient - in this case, exit the entire training loop
                    if (block == 0) & (len(set(self.loss_trn)) == 1):
                        self.stuck_in_zero_grad = True
                        break
            
            # Exit loop if in area of zero gradient
            if self.stuck_in_zero_grad:
                break
            
            # Load self.best_mdl (lowest validation loss on final iteration)
            # as self.mdl and compute test set performance
            self.mdl.load_state_dict(self.best_mdl.state_dict())
            
            self.test(block)
            if interm is not None:
                torch.save(self.mdl, self.fileloc_model + self.interm +f'_{block}.pt')
        
        # Record the final training time and rmse
        self.train_time = (time.perf_counter() - start_time)/60
        if self.blocked_crossval == True:
            self.rmse_final = np.average(self.loss_tst[:, 1])
        else:
            self.rmse_final = self.loss_tst[-1, 1]
        print(f'\n\nTraining complete. Training time = {self.train_time:.1f} minutes.')
        print(f'\nAverage model performance on test sets: RMSE = {self.rmse_final:.4f}')
    
    # Test the model
    def test(self, block):
        # Load testing data for the current block
        xy_tst = self.xy[block + 2, :, :, :].view((self.points_per_block,
                                                    self.total_steps,
                                                    self.total_dims)
                                                    )
                 
        # Remove the first total_steps examples to avoid data leakage
        x_tst = (xy_tst[self.total_steps:, :-self.output_steps, :]).to(device)
        # y_tst only needs effluent turbidity as teacher forcing is not used
        y_tst = (xy_tst[self.total_steps:, -self.output_steps:, 0]).to(device)
        
        # Create a "batch generator" that will load all the batches for
        # this iteration (epoch)
        batch_gen = func_batcher(self.batch_size,
                                 self.points_per_block - self.total_steps
                                 )
        
        # Empty the testing loss list
        loss_tst = []
        
        with torch.no_grad():
            # Iterate through all the validation batches in this epoch
            for bat in batch_gen:
                x_bat, y_bat = x_tst[bat, :, :], y_tst[bat, :]
                preds_tst = (self.mdl.forward(x_bat, y_bat)).to(device)
                rmse_tst = self.func_loss(preds_tst, y_bat.view((self.batch_size, self.output_steps, 1)))
                loss_tst.append(rmse_tst.tolist())
        
            # Save the average loss for this iteration to the testing loss array
            rmse_tst = np.mean(loss_tst)
            self.loss_tst[block, :] = (self.total_iters, rmse_tst)
            print(f'Model performance on test set (block {block + 3}): RMSE = {rmse_tst:.4f}')
    
    
    # Function to calculate loss in NTU
    def func_loss(self, preds, actual):
        loss = nn.MSELoss()
        turb_min, turb_max = 0, 10 # These are the minimum/maximum value of settled turbidity
                                   #  values on which the data was normalized (see data_torch_mgmt.py)
        rmse = torch.sqrt(loss((((turb_max - turb_min) * preds) + turb_min),
                               (((turb_max - turb_min) * actual) + turb_min))).to(device)
        return rmse
    
    
    # Save the model - this includes:
    #   - A .pt file of its parameters
    #   - A .txt file logging the characteristics of the model, its performance,
    #       and the characteristics of the optimizer
    #   - A .npy file of the training performance (RMSE) values for each epoch
    #   - A .png file of a plot of the training performance
    def save(self):
        modelname = (f"{self.model_prefix}_{self.num_layers}x{self.hidden_dims}_RMSE-{self.rmse_final:.4f}")
        
        # Save the model parameters as a .pt file
        torch.save(self.mdl, f"{self.fileloc_model}{modelname}.pt")
        
        # Save a log detailing the characteristics of the model, its performance,
        # and the characteristics of the optimizer as a .txt file
        with open(f"{self.fileloc_model}{modelname}.txt", 'w') as f:
            f.write(modelname)
            f.write(f"\nData set: {self.fileloc_data}{self.training_data_filename}")
            f.write(f"\nOutput model: {self.fileloc_model}{modelname}.pt")
            f.write("\n\nModel hyperparameters:")
            for param_name, param in zip(self.mdl_parameter_names, self.mdl_parameters):
                f.write(f"\n{param_name}: {param}")
            f.write(f"\n\nTraining time = {self.train_time:.1f} minutes on a {torch.cuda.get_device_name()} over {self.total_iters} iterations.")
            f.write(f"\nFinal model performance on test set: RMSE = {self.rmse_final:.4f}")
            f.write("\n\nOptimizer specifications:")
            for i in self.opt.defaults.items():
                item = str(i[0])
                value = self.opt.state_dict()['param_groups'][0][item]
                f.write(f"\n{item}: {value}")
            f.write(f"\n\nInterm: {self.interm}")
        
        # Save a plot of the training performance as a .png file
        fig = plt.figure(figsize=(10,6))
        plt.plot(range(1, self.total_iters + 1), self.loss_trn, 'r-')
        plt.plot(range(1, self.total_iters + 1), self.loss_val, 'g-')
        if self.blocked_crossval == True:
            plt.plot(self.loss_tst[:,0], self.loss_tst[:,1], 'bo', fillstyle='full')
            plt.plot([0, self.total_iters], [self.rmse_final, self.rmse_final], 'b-')
            for i in range(len(self.loss_tst)):
                plt.text(self.loss_tst[i,0] + 5, self.loss_tst[i,1], i + 3, fontsize=10)
        else:
            plt.plot(self.loss_tst[-1,0], self.loss_tst[-1,1], 'bo', fillstyle='full')
            plt.plot([0, self.total_iters], [self.rmse_final, self.rmse_final], 'b-')
            plt.text(self.loss_tst[-1,0] + 5, self.loss_tst[-1,1], len(self.loss_tst) + 2, fontsize=10)
        plt.title(modelname)
        plt.xlabel('Iterations')
        plt.ylabel('RMSE')
        #plt.ylim((0, 1))
        plt.legend(['Training', 'Validation', 'Test (block)', 'Test (mean)'])
        fig.set_dpi(100.)
        plt.savefig(f"{self.fileloc_model}{modelname}.png")
        
        # Save the training performance data as a .npy file
        np.save(f"{self.fileloc_model}{modelname}_perf",
                np.array((self.total_iters + 1,
                          self.loss_trn,
                          self.loss_val,
                          self.loss_tst,
                          self.rmse_final),
                         dtype=object),
                allow_pickle=True
                )
        
#%% MAIN

if __name__ == '__main__':  
        counter = 0
        m = 36 # This refers to the number of months in the training data;
               #  the training data files include this in their filename
        # Can train multiple models in one go
        # Layer, hidden are number of layers and hidden layer size respectively
        # Use_existing True/False decides if parameters are intialized from
        #  an existing model or randomly
        # Iterm is a filename prefix if you want to save a copy of the model
        #  after each block to see how model performance improves over the
        #  course of training
        for layer, hidden, use_existing, interm in [(3, 64, True, None),
                                                    (3, 64, True, None),
                                                    (3, 64, True, None)]:
            existing = "SP_4retraining.pt" if use_existing else None
            counter += 1
            trainer = rnn_trainer(fileloc_data, fileloc_model,
                                  model_prefix=f"RETR-{m}_frozen_{counter}", # arbitrary but descriptive
                                  existing_model=existing, interm=interm)
            trainer.initialize(
                               training_data_filename=f"stn_{m}_TRAIN_n.pt",
                               batch_size=64,
                               num_training_blocks=12-2,
                               hidden_dims=hidden,
                               num_layers=layer,
                               frozen_layers=2
                               )
            trainer.train(
                          learning_rate=5e-4,
                          max_iters_per_block=round(540/m),
                          max_iters_increasing_validation_loss=round(180/m),
                          weight_decay=1e-8,
                          gradient_clipping_norm=1,
                          blocked_crossval=True
                          )
            trainer.save()