# DataSize_DWT
Supplementary material for the article "Investigating Strategies for Reducing Dataset Size Requirements for Predicitive Machine Learning Models in Drinking Water Treatment".

See also [RNN_BCV_DWT](https://github.com/jakovaleks/RNN_BCV_DWT) for code from previous project.

## Description of the files in this repository:

- `data_import_mgmt.py` - For importing data from their original .csv files into Python-compatible formats.
- `data_cleaning_mgmt.py` - For removing anomalous data from the data set. See the comments in this file to see the criteria on which data were removed.
- `data_torch_mgmt.py` - For converting NumPy arrays into PyTorch tensors that are used for model training and evaluation.
- `models.py` - Contains the encoder-decoder GRU model.
- `train_rnn.py` - For training the model.
- `final_test.py` - For assessing the performance of trained models on the production data set.
- `data_rebalancing.py` - For rebalancing the data, as described in the paper.
- `_production_summary.txt` - Created in final_test.py, contains the RMSE and MAE of each model as evaluated on the production set.
