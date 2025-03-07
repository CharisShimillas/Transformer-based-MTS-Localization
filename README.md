
# Transformer-based MTS Anomaly Localization

## Paper
This is the code for the paper: Transformer-based Multivariate Time Series  Anomaly Localization.

You can get a free copy of the pre-print version from arXiv ([link](https://arxiv.org/abs/2501.08628)) .

Alternatively, you can get the published version from the publisher’s website (link).
## Instructions
You can train and test the model for each dataset using the command found in the scripts folder.

**To apply the model to your dataset**: Ensure that in your data folder, you have (1) train, (2) test data, and (3) test_labels datasets. The files can be in csv or npy format , but make sure to adjust the data_loader function accordingly to accommodate your file format. Rows correspond to time steps, while columns correspond to time-series dimensions. In the test label file, a '1' indicates that the corresponding time series (column) is anomalous at this time step (row), and normal otherwise. Make sure to adjust the 
<span style="color:green;">data_loader</span> function.

Please check the `instructions.txt` file for detailed usage instructions.

## Requirements
This project requires Python 3.7 or higher. Please see the `requirements.txt` file for a list of necessary libraries and packages.

## Citation Request
If you find our paper or any part of our code useful, please cite our work as follows:

