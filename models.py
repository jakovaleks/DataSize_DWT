# This file contains the GRU model class.

import torch
import random
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)


# =========================================================================== #
### ====== GRU ===== ###
class GRU(nn.Module):
    # Initialize an encoder-decoder GRU based on the given hyperparameters
    def __init__(self,
                 total_dims,
                 hidden_dims,
                 num_layers,
                 batch_size,
                 input_steps,
                 output_steps,
                 encoder_dropout_ratio,
                 decoder_dropout_ratio,
                 teacher_forcing_ratio,
                 ):
        super(GRU, self).__init__()
        
        # Initialize the encoder
        self.encoder = GRU_Encoder(total_dims,
                                   hidden_dims,
                                   num_layers,
                                   batch_size,
                                   encoder_dropout_ratio,
                                   )
        
        # Initialize the decoder
        self.decoder = GRU_Decoder(total_dims,
                                   hidden_dims,
                                   num_layers,
                                   decoder_dropout_ratio
                                   )
        
        # Save required hyperparameters
        self.total_dims = total_dims
        self.batch_size = batch_size
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.output_steps = output_steps
        self.teacher_forcing_ratio = teacher_forcing_ratio
        
        # Add a model description to print to the console
        self.desc = f'encoder-decoder GRU {total_dims}-{num_layers}x{hidden_dims}-1'
    
    # Forward pass on the data
    def forward(self, data_in, true_out):
        # Create empty predictions tensor
        preds = torch.zeros(self.batch_size, self.output_steps, 1).to(device)
        
        # Run forward pass on encoder
        # h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_dims).to(device)
        # nn.init.xavier_uniform_(h0)
        h1 = self.encoder.forward(data_in)#, h0)
        
        # Take last time step of the input as first input to the decoder
        x0 = data_in[:, -1, :].view((self.batch_size, 1, self.total_dims))
        
        # Loop through the decoder for the desired number of output steps
        for t in range(self.output_steps):
            # Run forward pass on the decoder for one time step
            data_out, pred, h1 = self.decoder.forward(x0, h1)
            
            # Add the predicted effluent turbidity to the predictions tensor
            preds[:, t, :] = pred
            
            # Roll to see if we will use teacher forcing for the next time step
            teacher_forcing_roll = (random.random() < self.teacher_forcing_ratio)
            
            # Set the next input to the decoder (either the previous prediction
            # or the ground truth, depending on teacher_forcing_roll)
            if teacher_forcing_roll:
                x0 = true_out[:, t, :].view((self.batch_size, 1, self.total_dims))
            else:
                x0 = data_out
        
        # Return predicted effluent turbidity
        return preds

# Encoder module for GRU
class GRU_Encoder(nn.Module):
    # Initialize the encoder
    def __init__(self,
                 total_dims,
                 hidden_dims,
                 num_layers,
                 batch_size,
                 encoder_dropout_ratio,
                 ):
        super(GRU_Encoder, self).__init__()
        
        # Encoder GRU layer
        self.gru = nn.GRU(input_size=total_dims,
                          hidden_size=hidden_dims,
                          num_layers=num_layers, 
                          batch_first=True,
                          dropout=encoder_dropout_ratio,
                          device=device
                          )
        
        # Input dropout layer
        self.dropout = nn.Dropout(encoder_dropout_ratio)

    # Encoder forward pass on the data
    def forward(self, data_in):#, h0):
        # Apply dropout
        data_in = self.dropout(data_in)
        
        # Pass through GRU layer
        data_out, h1 = self.gru(data_in)#, h0)
        
        # Return hidden state 
        return h1

# Decoder module for GRU
class GRU_Decoder(nn.Module):
    # Initialize the decoder
    def __init__(self,
                 total_dims,
                 hidden_dims,
                 num_layers,
                 decoder_dropout_ratio
                 ):
        super(GRU_Decoder, self).__init__()
        
        # Decoder GRU layer
        self.gru = nn.GRU(input_size=total_dims,
                          hidden_size=hidden_dims,
                          num_layers=num_layers, 
                          batch_first=True,
                          dropout=decoder_dropout_ratio,
                          device=device
                          )
        
        # Dropout layers
        self.dropout0 = nn.Dropout(decoder_dropout_ratio)
        self.dropout1 = nn.Dropout(decoder_dropout_ratio)
        
        # Decoder output layer
        self.affine = nn.Linear(hidden_dims, total_dims, device=device)
     
    # Decoder forward pass on the data
    def forward(self, data_in, h1):
        # Apply dropout
        data_in = self.dropout0(data_in)

        # Pass through GRU layers using hidden and cell states from the encoder        
        data_out, h1 = self.gru(data_in, h1)
        
        # Pass through output layers
        data_out = self.dropout1(data_out)
        data_out = self.affine(data_out)
        
        # Return only the effluent turbidity (column 0) as a prediction
        pred = data_out[:, :, 0]
        
        # Return the output, prediction, and hidden and cell states
        return data_out, pred, h1

# =========================================================================== #
