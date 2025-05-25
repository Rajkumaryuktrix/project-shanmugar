import numpy as np
import pandas as pd
from typing import List, Optional
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class ForwardTesting:
    def __init__(self, data: pd.DataFrame, base_price: str):
        """
        Initialize forward testing class.
        
        Args:
            data: Historical price data
            base_price: Column name for base price (e.g., 'close')
        """
        self.data = data
        self.base_price = base_price
        self.future_data_frames: List[pd.DataFrame] = []
        self.generate_future_prices()

    def generate_future_prices(self) -> None:
        """Generate future price scenarios using vectorized operations."""
        # Calculate daily returns using vectorized operations
        daily_returns = self.data[self.base_price].pct_change().dropna().values
        
        # Calculate mean and standard deviation of daily returns
        mean_return = np.mean(daily_returns)
        std_dev = np.std(daily_returns)
        
        # Generate three scenarios with different volatility levels
        volatility_multipliers = [0.10, 1.0, 2.0]
        
        for multiplier in volatility_multipliers:
            # Set random seed for reproducibility
            np.random.seed(volatility_multipliers.index(multiplier))
            
            # Create a copy of the original data
            future_df = self.data.copy()
            
            # Generate future returns using normal distribution
            future_returns = np.random.normal(
                loc=mean_return,
                scale=std_dev * multiplier,
                size=len(future_df)
            )
            
            # Calculate future prices using vectorized operations
            future_prices = future_df[self.base_price].iloc[-1] * (1 + future_returns).cumprod()
            
            # Update the price column in the future data frame
            future_df[self.base_price] = future_prices
            
            # Append to list of future scenarios
            self.future_data_frames.append(future_df)
            
            logger.info(f"Generated future price scenario with volatility multiplier {multiplier}")
            
    def get_scenario_statistics(self, scenario_index: int) -> dict:
        """
        Calculate statistics for a specific future scenario.
        
        Args:
            scenario_index: Index of the future scenario
            
        Returns:
            Dictionary containing scenario statistics
        """
        if not 0 <= scenario_index < len(self.future_data_frames):
            raise ValueError(f"Invalid scenario index. Must be between 0 and {len(self.future_data_frames)-1}")
            
        future_df = self.future_data_frames[scenario_index]
        
        # Calculate basic statistics using vectorized operations
        returns = future_df[self.base_price].pct_change().dropna().values
        
        stats = {
            'mean_return': np.mean(returns),
            'std_dev': np.std(returns),
            'min_price': np.min(future_df[self.base_price]),
            'max_price': np.max(future_df[self.base_price]),
            'final_price': future_df[self.base_price].iloc[-1],
            'price_change_pct': ((future_df[self.base_price].iloc[-1] / future_df[self.base_price].iloc[0]) - 1) * 100
        }
        
        return stats
        
    def get_all_scenarios_statistics(self) -> pd.DataFrame:
        """
        Calculate statistics for all future scenarios.
        
        Returns:
            DataFrame containing statistics for all scenarios
        """
        stats_list = []
        
        for i in range(len(self.future_data_frames)):
            scenario_stats = self.get_scenario_statistics(i)
            scenario_stats['scenario'] = i
            stats_list.append(scenario_stats)
            
        return pd.DataFrame(stats_list)
        
    def plot_scenario(self, scenario_index: int) -> None:
        """
        Plot a specific future scenario.
        
        Args:
            scenario_index: Index of the future scenario
        """
        if not 0 <= scenario_index < len(self.future_data_frames):
            raise ValueError(f"Invalid scenario index. Must be between 0 and {len(self.future_data_frames)-1}")
            
        future_df = self.future_data_frames[scenario_index]
        
        # Plot the scenario
        plt.figure(figsize=(12, 6))
        plt.plot(future_df.index, future_df[self.base_price])
        plt.title(f'Future Price Scenario {scenario_index}')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.grid(True)
        plt.show()
        
    def plot_all_scenarios(self) -> None:
        """Plot all future scenarios."""
        plt.figure(figsize=(12, 6))
        
        for i, future_df in enumerate(self.future_data_frames):
            plt.plot(future_df.index, future_df[self.base_price], label=f'Scenario {i}')
            
        plt.title('All Future Price Scenarios')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show() 