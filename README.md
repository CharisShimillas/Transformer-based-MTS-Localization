
# Transformer-based MTS Anomaly Localization

## Paper
This is the code for the paper: Transformer-based Multivariate Time Series  Anomaly Localization.

You can get a free copy of the pre-print version from arXiv ([link](https://arxiv.org/abs/2501.08628)) .

Alternatively, you can get the published version from the publisher’s website (link).
## Instructions

You can train and test the model for each dataset using the command found in the scripts folder.

### Example for the ASD Dataset

To train and test the model on the ASD dataset, use the following commands:

```bash
# Train the model
python main.py --anormly_ratio 1 --num_epochs 1 --batch_size 128 --mode train --dataset ASD_ALL --data_path path_to_data_folder --input_c 19 --output_c 19

# Test the model
python main.py --anormly_ratio 1 --num_epochs 1 --batch_size 128 --mode test --dataset ASD_ALL --data_path path_to_data_folder --input_c 19 --output_c 19
```

### To apply the model to your dataset: 
Ensure that in your data folder, you have:
- **Train Data**: Training dataset.
- **Test Data**: Testing dataset.
- **Test Labels**: Labels indicating anomalies ('1' for anomalies, '0' for normal).

Rows correspond to time steps, while columns correspond to time-series dimensions. In the test label file, a '1' in the ij cell indicates that the j-th time series is anomalous at time step i.

The files can be in csv or npy format , but make sure to adjust  the ***data_loader*** function.

Please check the `instructions.txt` file for detailed usage instructions.

## Requirements
This project requires Python 3.7 or higher. Please see the `requirements.txt` file for a list of necessary libraries and packages.

## Citation Request
If you find our paper or any part of our code useful, please cite our work as follows:

