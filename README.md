
# Transformer-based MTS Anomaly Localization
With the growing complexity of Cyber-Physical
Systems (CPS) and the integration of Internet of Things (IoT),
the use of sensors for online monitoring generates large volume of
multivariate time series (MTS) data. Consequently, the need for
robust anomaly diagnosis in MTS is paramount to maintaining
system reliability and safety. While significant advancements have
been made in anomaly detection, localization remains a largely
underexplored area, though crucial for intelligent decisionmaking. This paper introduces a novel transformer-based model
for unsupervised anomaly diagnosis in MTS, with a focus on
improving localization performance, through an in-depth analysis
of the self-attention mechanism’s learning behavior under both
normal and anomalous conditions. We formulate the anomaly
localization problem as a three-stage process: time-step, window,
and segment-based. This leads to the development of the SpaceTime Anomaly Score (STAS), a new metric inspired by the
connection between transformer latent representations and spacetime statistical models. STAS is designed to capture individual anomaly behaviors and inter-series dependencies, delivering
enhanced localization performance. Additionally, the Statistical
Feature Anomaly Score (SFAS) complements STAS by analyzing
statistical features around anomalies, with their combination
helping to reduce false alarms. Experiments on real world and
synthetic datasets illustrate the model’s superiority over state-ofthe-art methods in both detection and localization tasks.
## Paper
This is the code for the paper: Transformer-based Multivariate Time Series  Anomaly Localization.

You can get a free copy of the pre-print version from arXiv ([link](https://arxiv.org/abs/2501.08628)) .

Alternatively, you can get the published version from the publisher’s website (link).

## Datasets:
For the SMD and ASD datasets, please access them through the following repositories, which are associated with their respective research works:

- **SMD**: The dataset is available in the repository related to the KDD 2019 paper "Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network". [View Repository](https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset).
- **ASD**: The dataset can be found in the [InterFusion project on GitHub](https://github.com/zhhlee/InterFusion/tree/main).

### To apply the model to your dataset: 
Ensure that in your data folder, you have:
- **Train Data**: Training dataset.
- **Test Data**: Testing dataset.
- **Test Labels**: Labels indicating anomalies ('1' for anomalies, '0' for normal).

Rows correspond to time steps, while columns correspond to time-series dimensions. In the test label file, a '1' in the ij cell indicates that the j-th time series is anomalous at time step i.

The files can be in csv or npy format , but make sure to adjust  the ***data_loader*** function.

## Training & Evaluating
You can train and test the model for each dataset using the command found in the scripts folder.


Please check the `instructions.txt` file for detailed usage instructions.

## Requirements
This project requires Python 3.7 or higher. Please see the `requirements.txt` file for a list of necessary libraries and packages.

## Citation Request
If you find our paper or any part of our code useful, please cite our work as follows:

