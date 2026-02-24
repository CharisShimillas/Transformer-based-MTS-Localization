import numpy as np
import pandas as pd
import os

def compute_anomaly_scores(base_path, dataset_name="SMD", num_features=38, correlation_type="spearman"):
    """
    Computes anomaly scores using the anomaly score CSVs from different training runs.
    
    Parameters:
        base_path (str): The base directory containing the anomaly score CSVs.
        dataset_name (str): The main dataset name.
        num_features (int): The number of time series features.
        correlation_type (str): Type of correlation matrix ("pearson", "spearman", "kendall").
    
    Returns:
        None (saves the computed anomaly scores to a CSV file).
    """
    
    # Load the total anomaly score file with proper column names
    total_anomaly_score_all = pd.read_csv(
        os.path.join(base_path, dataset_name, "anomaly_scores.csv"), 
        header=0  # Read first row as column names
    )
    total_anomaly_score_all.columns = ["Timestamp", "Anomaly_Score"]
    
    print("Columns:", total_anomaly_score_all.columns)
    print("First few rows:\n", total_anomaly_score_all.head())
    
    # Load the Anomalys score from masking process
    anomaly_scores_excluding = []
    for i in range(num_features):
        path = os.path.join(base_path, f"{dataset_name}{i}", "anomaly_scores.csv")
        df = pd.read_csv(path, header=0)
        df.columns = ["Timestamp", "Anomaly_Score"]
        anomaly_scores_excluding.append(df)
    
    # Load the correlation matrix
    correlation_matrix = np.abs(np.load(r'/spearman_correlation_matrix.npy'))

   # Initialize a DataFrame to store the anomaly scores
    anomaly_scores_df = pd.DataFrame({'Timestamp': total_anomaly_score_all["Timestamp"]})
    
    # Compute anomaly scores for each feature
    for i in range(num_features):
        total_anomaly_score_excluding_i = anomaly_scores_excluding[i]["Anomaly_Score"]
        total_anomaly_score_all_values = total_anomaly_score_all["Anomaly_Score"]
        
        # Compute weighted sum of reconstruction errors excluding feature i
        weighted_sum = np.zeros_like(total_anomaly_score_all_values)
        for j in range(num_features):
            if j != i:
                total_anomaly_score_excluding_j = anomaly_scores_excluding[j]["Anomaly_Score"]
                weighted_sum += correlation_matrix[i, j] * (total_anomaly_score_all_values - total_anomaly_score_excluding_j)**2
                
        # Final anomaly score computation
        anomaly_scores = ((total_anomaly_score_all_values - total_anomaly_score_excluding_i)**2) + weighted_sum
        anomaly_scores_df[f'Anomaly Score Time Series {i+1}'] = anomaly_scores.values
    
    # Save the final anomaly scores to a CSV
    output_path = os.path.join(base_path, dataset_name, "final_anomaly_scores.csv")
    anomaly_scores_df.to_csv(output_path, index=False)
    print(f"Anomaly scores saved to {output_path}")
