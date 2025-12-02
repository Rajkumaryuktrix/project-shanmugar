import numpy as np
import pandas as pd
from typing import Tuple, Dict
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import chi2, ks_2samp, jarque_bera
import matplotlib.pyplot as plt
import logging
import os
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class AdvancedStatisticalTesting:
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize statistical testing class.
        
        Args:
            significance_level: Significance level for statistical tests (default: 0.05)
        """
        self.significance_level = significance_level
        self.EPS = 1e-8  # Small threshold for numerical stability

    def ljung_box_test(self, time_series: np.ndarray, lags: int = 10) -> Tuple[float, Dict]:
        """
        Perform Ljung-Box test for autocorrelation.
        
        Args:
            time_series: Array of time series data
            lags: Number of lags to test
            
        Returns:
            Tuple of (test statistic, results dictionary)
        """
        # Convert to numpy array if not already
        time_series = np.array(time_series)
        
        # Calculate sample size
        sample_size = len(time_series)
        
        # Perform Ljung-Box test
        results = acorr_ljungbox(time_series, lags=lags)
        p_values = results['lb_pvalue']
        
        # Calculate standard errors and test statistics
        lr_standard_errors = []
        lr_test_statistics = []
        
        for lag, p_value in enumerate(p_values, start=1):
            # Calculate test statistic
            lr_test_statistic = -2 * np.log(p_value)
            lr_test_statistics.append(lr_test_statistic)
            
            # Calculate standard error
            ljung_box_standard_error = np.sqrt(2 * sample_size * (sample_size + 2) * lr_test_statistic) / (sample_size * (sample_size + 2) - 2 * sample_size)
            lr_standard_errors.append(ljung_box_standard_error)
            
            # Calculate critical value
            critical_value = chi2.ppf(1 - self.significance_level, df=lag)
            
            # Log results
            logger.info(f"Lag {lag}:")
            logger.info(f"  p-value = {p_value:.4f}")
            logger.info(f"  Test Statistic = {lr_test_statistic:.4f}")
            logger.info(f"  Standard Error = {ljung_box_standard_error:.4f}")
            
            if lr_test_statistic > critical_value:
                logger.info(f"  Reject the null hypothesis of no autocorrelation at Lag {lag}")
            else:
                logger.info(f"  Fail to reject the null hypothesis of no autocorrelation at Lag {lag}")
        
        return lr_test_statistics[-1], results

    def kolmogorov_smirnov_test(self, condition_1: np.ndarray, condition_2: np.ndarray) -> Tuple[float, float]:
        """
        Perform Kolmogorov-Smirnov test for comparing distributions.
        
        Args:
            condition_1: First sample array
            condition_2: Second sample array
            
        Returns:
            Tuple of (test statistic, p-value)
        """
        # Convert to numpy arrays if not already
        condition_1 = np.array(condition_1)
        condition_2 = np.array(condition_2)
        
        # Perform KS test
        ks_statistic, ks_p_value = ks_2samp(condition_1, condition_2)
        
        # Log results
        logger.info(f"KS Statistic: {ks_statistic:.4f}")
        logger.info(f"P-value: {ks_p_value:.4f}")
        
        if ks_p_value > self.significance_level:
            logger.info("The distributions may be similar (fail to reject the null hypothesis).")
        else:
            logger.info("The distributions are likely different (reject the null hypothesis).")
            
        return ks_statistic, ks_p_value

    def jarque_bera_test(self, strategy_returns: np.ndarray) -> Tuple[float, float]:
        """
        Perform Jarque-Bera test for normality.
        
        Args:
            strategy_returns: Array of strategy returns
            
        Returns:
            Tuple of (test statistic, p-value)
        """
        # Convert to numpy array if not already
        strategy_returns = np.array(strategy_returns)
        
        # Perform Jarque-Bera test
        jb_statistic, jb_p_value = jarque_bera(strategy_returns)
        
        # Log results
        logger.info(f"Jarque-Bera Statistic: {jb_statistic:.4f}")
        logger.info(f"P-value: {jb_p_value:.4f}")
        
        if jb_p_value > self.significance_level:
            logger.info("The returns may follow a normal distribution (fail to reject the null hypothesis).")
        else:
            logger.info("The returns are likely not normally distributed (reject the null hypothesis).")
            
        return jb_statistic, jb_p_value

    def calculate_correlations(self, series1: np.ndarray, series2: np.ndarray, max_lag: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate auto-correlation and cross-correlation.
        
        Args:
            series1: First time series
            series2: Second time series
            max_lag: Maximum lag to calculate
            
        Returns:
            Tuple of (auto-correlation array, cross-correlation array)
        """
        # Convert to numpy arrays if not already
        series1 = np.array(series1)
        series2 = np.array(series2)
        
        # Calculate correlations using numpy's correlate function
        auto_corr = np.correlate(series1, series1, mode='full') / np.sum(series1**2)
        cross_corr = np.correlate(series1, series2, mode='full') / np.sqrt(np.sum(series1**2) * np.sum(series2**2))
        
        # Keep only positive lags
        auto_corr = auto_corr[len(auto_corr)//2:]
        cross_corr = cross_corr[len(cross_corr)//2:]
        
        # Save plot data to file instead of plotting
        plots_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'results', 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_file = os.path.join(plots_dir, f'correlations_{timestamp}.json')
        plot_data = {
            "title": "Correlation Analysis",
            "series": [
                {"label": "Auto-correlation", "x": list(range(max_lag+1)), "y": auto_corr[:max_lag+1].tolist()},
                {"label": "Cross-correlation", "x": list(range(max_lag+1)), "y": cross_corr[:max_lag+1].tolist()}
            ],
            "xlabel": "Lag",
            "ylabel": "Correlation"
        }
        with open(plot_file, 'w') as f:
            json.dump(plot_data, f, indent=4)
        return auto_corr, cross_corr

    def hurst_exponent(self, ts: np.ndarray) -> float:
        """
        Calculate Hurst exponent for time series.
        
        Args:
            ts: Time series array
            
        Returns:
            Hurst exponent value
        """
        # Convert to numpy array if not already
        ts = np.array(ts)
        
        # Calculate lags and tau
        lags = range(2, 20)
        tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
        
        # Calculate Hurst exponent using linear regression
        hurst_exponent = np.polyfit(np.log(lags), np.log(tau), 1)[0]
        
        # Log interpretation
        if hurst_exponent > 0.5:
            logger.info("The series exhibits a trending behavior (Hurst exponent > 0.5).")
        elif hurst_exponent < 0.5:
            logger.info("The series exhibits a mean-reverting behavior (Hurst exponent < 0.5).")
        else:
            logger.info("The series exhibits a random walk behavior (Hurst exponent = 0.5).")
            
        # Save plot data to file instead of plotting
        plots_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'results', 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_file = os.path.join(plots_dir, f'hurst_{timestamp}.json')
        plot_data = {
            "title": "Hurst Exponent Calculation",
            "series": [
                {"label": "Time Series", "x": list(range(len(ts))), "y": ts.tolist()},
                {"label": "Log-Log Plot", "x": np.log(list(range(2, 20))).tolist(), "y": np.log(tau).tolist()}
            ],
            "xlabel": "Index",
            "ylabel": "Value"
        }
        with open(plot_file, 'w') as f:
            json.dump(plot_data, f, indent=4)
        return hurst_exponent 