"""
Moving Average Crossover Strategy implementation.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class MovingAverageCrossoverStrategy:
    def __init__(self, short_window=20, long_window=50):
        """
        Initialize the Moving Average Crossover Strategy.
        
        Args:
            short_window (int): Window size for short-term moving average
            long_window (int): Window size for long-term moving average
        """
        self.short_window = short_window
        self.long_window = long_window
        self.signals = pd.DataFrame()
        self.last_signal = None

    def generate_signals(self, data):
        """
        Generate trading signals based on moving average crossover.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with signals
        """
        # Calculate moving averages
        data['SMA_short'] = data['close'].rolling(window=self.short_window).mean()
        data['SMA_long'] = data['close'].rolling(window=self.long_window).mean()
        
        # Initialize signal column with None
        data['Signal'] = None
        
        # Generate signals
        data.loc[data['SMA_short'] > data['SMA_long'], 'Signal'] = "Buy"
        data.loc[data['SMA_short'] < data['SMA_long'], 'Signal'] = "Sell"
        
        # Forward fill signals
        data['Signal'] = data['Signal'].ffill()
        
        # Log signal generation results
        logger.info(f"Generated signals for {len(data)} data points")
        logger.info(f"Unique signals: {data['Signal'].unique()}")
        logger.info(f"Signal counts:\n{data['Signal'].value_counts()}")
        
        return data 