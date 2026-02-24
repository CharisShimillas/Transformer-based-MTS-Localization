
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os

# Statistical features functions
def lumpiness(x, width=10):
    start = np.arange(0, len(x), width)
    end = np.append(start[1:] - 1, len(x) - 1)
    varx = [np.var(x[start[i]:end[i]+1], ddof=1) for i in range(len(start))]
    lumpiness = np.var(varx, ddof=1)
    return lumpiness

def level_shift(x, width=10):
    start = np.arange(0, len(x) - width + 1, width)
    means = [np.mean(x[i:i+width]) for i in start]
    lshifts = np.max(np.abs(np.diff(means)), initial=0)
    return lshifts

def variance_change(x, width=10):
    roll_var = pd.Series(x).rolling(window=width).var(ddof=1)
    if roll_var.isnull().all():
        return np.nan
    vchange = np.max(np.abs(np.diff(roll_var.dropna())))
    return vchange

def linearity(x):
    if len(x) < 2:
        return np.nan
    p = np.polyfit(range(len(x)), x, 1)
    return p[0]

def curvature(x):
    if len(x) < 3:
        return np.nan
    p = np.polyfit(range(len(x)), x, 2)
    return p[0]

def spikiness(x, width=10):
    roll_mean = pd.Series(x).rolling(window=width).mean()
    if roll_mean.isnull().all():
        return np.nan
    return np.var(x - roll_mean.dropna(), ddof=1)

def burstiness(x):
    if np.mean(x) == 0:
        return np.nan
    return np.var(x, ddof=1) / np.mean(x)

def rmeaniqmean(x):
    iqmean = np.mean(np.sort(x)[len(x)//4:3*len(x)//4])
    return iqmean / np.mean(x)

def moment3(x):
    return np.mean((x - np.mean(x))**3) / np.std(x)**3

def highlowmu(x):
    mu = np.mean(x)
    if mu == 0:
        return np.nan
    mhi = np.mean(x[x > mu])
    mlo = np.mean(x[x < mu])
    return (mhi - mu) / (mu - mlo)
def skewness(x):
    return pd.Series(x).skew()

def kurtosis(x):
    return pd.Series(x).kurtosis()

def energy(x):
    return np.sum(np.square(x))

def zero_crossing_rate(x):
    return np.sum(np.diff(np.sign(x)) != 0) / len(x)

def ASSfas(config):
    """
    Compute the SFAS anomaly score for a dataset.
    
    Args:
        config: Configuration object containing paths and parameters.

    Returns:
        A DataFrame containing SFAS anomaly scores for each anomalous time step.
    """
    # set paths from config
    test_data_path = os.path.join(config.data_path, 'test.csv')
    test_label_path = os.path.join(config.data_path, 'test_label.csv')

    print(f"Loading test data from: {test_data_path}")
    print(f"Loading test labels from: {test_label_path}")

    # Load test data
    df_test = pd.read_csv(test_data_path)
    df_test_labels = pd.read_csv(test_label_path)

    # # Add a 'time_index' column
    # df_test.insert(0, 'time_index', range(len(df_test)))
    # df_test_labels.insert(0, 'time_index', range(len(df_test_labels)))
    df_test_labels.rename(columns={'0': 'anomaly'}, inplace=True)

    # Get the list of anomalous time steps from the labels
    if 'anomaly' in df_test_labels.columns and 'time_index' in df_test_labels.columns:
        anomalous_time_steps = df_test_labels[df_test_labels['anomaly'] == 1]['time_index'].values
    else:
        print("Error: 'anomaly' or 'time_index' column not found in test labels file.")
        return None

    def extract_features(df, width=20):
        features = pd.DataFrame()
        features['mean'] = df.apply(np.mean, axis=0)
        features['variance'] = df.apply(np.var, axis=0)
        features['lshift'] = df.apply(lambda x: level_shift(x, width), axis=0)
        features['vchange'] = df.apply(lambda x: variance_change(x, width), axis=0)
        features['linearity'] = df.apply(lambda x: linearity(x), axis=0)
        features['curvature'] = df.apply(lambda x: curvature(x), axis=0)
        features['spikiness'] = df.apply(lambda x: spikiness(x, width), axis=0)
        features['burstiness'] = df.apply(lambda x: burstiness(x), axis=0)
        features['highlowmu'] = df.apply(lambda x: highlowmu(x), axis=0)
        features['skewness'] = df.apply(lambda x: skewness(x), axis=0)
        features['kurtosis'] = df.apply(lambda x: kurtosis(x), axis=0)
        return features

    def identify_intervals(anomalous_time_steps):
        intervals = []
        if len(anomalous_time_steps) == 0:
            return intervals

        current_interval = [anomalous_time_steps[0]]
        for i in range(1, len(anomalous_time_steps)):
            if anomalous_time_steps[i] == anomalous_time_steps[i - 1] + 1:
                current_interval.append(anomalous_time_steps[i])
            else:
                intervals.append(current_interval)
                current_interval = [anomalous_time_steps[i]]
        intervals.append(current_interval)
        
        return intervals

    def compute_anomaly_scores(df, anomalous_time_steps, window_size=200):
        anomaly_scores = []
        intervals = identify_intervals(anomalous_time_steps)
        
        for interval in intervals:
            interval_start = interval[0]
            start = max(0, interval_start - window_size - 1)
            end = interval_start
            
            before_interval_df = df.iloc[start:end, 1:]  # Exclude time_index column
            before_interval_features = extract_features(before_interval_df)
            
            imputer = SimpleImputer(strategy='mean')
            before_interval_features_imputed = imputer.fit_transform(before_interval_features)
            
            scaler = StandardScaler()
            before_interval_features_scaled = scaler.fit_transform(before_interval_features_imputed)
            
            pca = PCA(n_components=2)
            before_interval_features_pca = pca.fit_transform(before_interval_features_scaled)
            
            for anomaly_time_step in interval:
                start = max(0, anomaly_time_step - 20)
                end = min(len(df), anomaly_time_step + 20 + 1)
                
                window_df = df.iloc[start:end, 1:]
                window_features = extract_features(window_df)
                
                window_features_imputed = imputer.fit_transform(window_features)
                window_features_scaled = scaler.transform(window_features_imputed)
                window_features_pca = pca.transform(window_features_scaled)
                
                for i in range(len(window_features_pca)):
                    score = np.linalg.norm(window_features_pca[i] - before_interval_features_pca[-1])
                    anomaly_scores.append((anomaly_time_step, score))
                    
        return anomaly_scores

    # Compute anomaly scores
    anomaly_scores = compute_anomaly_scores(df_test, anomalous_time_steps)
    print("Anomaly scores computed successfully.")

    # Convert to DataFrame
    df = pd.DataFrame(anomaly_scores, columns=['anomaly_index', 'value'])

    # Pivot the DataFrame
    df_pivot = df.pivot_table(index='anomaly_index', columns=df.groupby('anomaly_index').cumcount(), values='value')
    df_pivot.columns = [f'AS_TS{i}' for i in df_pivot.columns]
    df_pivot.reset_index(inplace=True)

    # Save results
    save_path = os.path.join(config.results_path, 'SFAS_anomaly_scores.csv')
    df_pivot.to_csv(save_path, index=False)
    
    print(f"Anomaly scores saved to {save_path}")

    return df_pivot

