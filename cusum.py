import numpy as np

def cusum(data, threshold, drift):
    """
    Applies the CUSUM algorithm to detect changes in a data series.

    Args:
    - data (np.array): Input data series.
    - threshold (float): The threshold value for detecting a change.
    - drift (float): The allowable drift.

    Returns:
    - np.array: An array with 1 indicating a change and 0 indicating no change.
    """
    s_pos = np.zeros(data.shape)
    s_neg = np.zeros(data.shape)
    changes = np.zeros(data.shape)

    for i in range(1, len(data)):
        s_pos[i] = max(0, s_pos[i-1] + data[i] - drift)
        s_neg[i] = max(0, s_neg[i-1] - data[i] - drift)
        if s_pos[i] > threshold or s_neg[i] > threshold:
            changes[i] = 1
            s_pos[i] = 0
            s_neg[i] = 0

    return changes
import numpy as np

def fir_cusum(data, threshold, drift, headstart):
    """
    Applies the FIR CUSUM algorithm to detect changes in a data series.

    Args:
    - data (np.array): Input data series.
    - threshold (float): The threshold value for detecting a change.
    - drift (float): The allowable drift.
    - headstart (float): The headstart value for faster initial response.

    Returns:
    - np.array: An array with 1 indicating a change and 0 indicating no change.
    """
    s_pos = np.zeros(data.shape)
    s_neg = np.zeros(data.shape)
    changes = np.zeros(data.shape)

    # Initialize the cumulative sums with the headstart value
    s_pos[0] = headstart
    s_neg[0] = headstart

    for i in range(1, len(data)):
        s_pos[i] = max(0, s_pos[i-1] + data[i] - drift)
        s_neg[i] = max(0, s_neg[i-1] - data[i] - drift)
        if s_pos[i] > threshold or s_neg[i] > threshold:
            changes[i] = 1
            s_pos[i] = 0
            s_neg[i] = 0

    return changes


# cusum_mean_detector.py

import torch
import numpy as np
from typing import Tuple

class CusumMeanDetector:
    def __init__(self, t_warmup=30, p_limit=0.01) -> None:
        self._t_warmup = t_warmup
        self._p_limit = p_limit
        self._reset()

    def predict_next(self, y: torch.tensor) -> Tuple[float, bool]:
        self._update_data(y)
        if self.current_t == self._t_warmup:
            self._init_params()
        if self.current_t >= self._t_warmup:
            prob, is_changepoint = self._check_for_changepoint()
            if is_changepoint:
                self._reset()
            return (1 - prob), is_changepoint
        else:
            return 0, False

    def _reset(self) -> None:
        self.current_t = torch.zeros(1)
        self.current_obs = []
        self.current_mean = None
        self.current_std = None

    def _update_data(self, y: torch.tensor) -> None:
        self.current_t += 1
        self.current_obs.append(y.reshape(1))

    def _init_params(self) -> None:
        self.current_mean = torch.mean(torch.concat(self.current_obs))
        self.current_std = torch.std(torch.concat(self.current_obs))

    def _check_for_changepoint(self) -> Tuple[float, bool]:
        standardized_sum = torch.sum(torch.concat(self.current_obs) - self.current_mean) / (self.current_std * self.current_t ** 0.5)
        prob = float(self._get_prob(standardized_sum).detach().numpy())
        return prob, prob < self._p_limit

    def _get_prob(self, y: torch.tensor) -> bool:
        p = torch.distributions.normal.Normal(0, 1).cdf(torch.abs(y))
        prob = 2 * (1 - p)
        return prob
