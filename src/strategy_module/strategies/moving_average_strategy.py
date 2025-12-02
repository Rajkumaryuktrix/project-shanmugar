"""
Moving Average Crossover Strategy implementation.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

class MovingAverageStrategy:
    """
    Moving Average Crossover Strategy for trading.
    Generates signals based on the crossover of two moving averages.
    """
    
    def __init__(self, fast_period=20, slow_period=50, sl_atr_multiplier=2, tp_atr_multiplier=3, atr_period=14):
        """
        Initialize Moving Average Strategy parameters.
        
        Args:
            fast_period (int): Period for fast moving average
            slow_period (int): Period for slow moving average
            sl_atr_multiplier (float): Multiplier for ATR to set stop loss
            tp_atr_multiplier (float): Multiplier for ATR to set take profit
            atr_period (int): Period for ATR calculation
        """
        # Convert parameters to appropriate types
        self.fast_period = int(fast_period)
        self.slow_period = int(slow_period)
        self.sl_atr_multiplier = float(sl_atr_multiplier)
        self.tp_atr_multiplier = float(tp_atr_multiplier)
        self.atr_period = int(atr_period)
        
        # Validate parameters
        self._validate_parameters()
        
        logger.info(f"Moving Average Strategy initialized with parameters:")
        logger.info(f"Fast Period: {self.fast_period}, Slow Period: {self.slow_period}")
        logger.info(f"SL ATR Multiplier: {self.sl_atr_multiplier}")
        logger.info(f"TP ATR Multiplier: {self.tp_atr_multiplier}")
        logger.info(f"ATR Period: {self.atr_period}")

    def _validate_parameters(self):
        """Validate strategy parameters"""
        if self.fast_period <= 0:
            raise ValueError("Fast period must be greater than 0")
        if self.slow_period <= 0:
            raise ValueError("Slow period must be greater than 0")
        if self.fast_period >= self.slow_period:
            raise ValueError("Fast period must be less than slow period")
        if self.sl_atr_multiplier <= 0:
            raise ValueError("Stop loss ATR multiplier must be greater than 0")
        if self.tp_atr_multiplier <= 0:
            raise ValueError("Take profit ATR multiplier must be greater than 0")
        if self.atr_period <= 0:
            raise ValueError("ATR period must be greater than 0")

    def calculate_atr(self, high, low, close):
        """Calculate Average True Range using vectorized operations"""
        try:
            # Calculate True Range components
            high_low = high - low
            high_close = np.abs(high - np.roll(close, 1))
            low_close = np.abs(low - np.roll(close, 1))
            
            # Calculate True Range
            true_range = np.maximum(np.maximum(high_low, high_close), low_close)
            
            # Calculate ATR using rolling window
            atr = np.zeros_like(close)
            for i in range(len(close)):
                if i < self.atr_period:
                    atr[i] = np.mean(true_range[:i+1])
                else:
                    atr[i] = np.mean(true_range[i-self.atr_period+1:i+1])
            
            return atr
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {str(e)}")
            raise

    def calculate_moving_average(self, data, window):
        """Calculate moving average using vectorized operations"""
        ma = np.zeros_like(data)
        for i in range(len(data)):
            if i < window:
                ma[i] = np.mean(data[:i+1])
            else:
                ma[i] = np.mean(data[i-window+1:i+1])
        return ma

    def generate_signals(self, data):
        """
        Generate trading signals based on moving average crossovers.
        
        Args:
            data (dict): Dictionary containing OHLCV data with keys:
                - time: array of timestamps
                - open: array of open prices
                - high: array of high prices
                - low: array of low prices
                - close: array of close prices
                - volume: array of volumes
            
        Returns:
            dict: Dictionary containing signals and SL/TP levels
        """
        try:
            # Validate input data
            required_keys = ['time', 'open', 'high', 'low', 'close', 'volume']
            if not all(key in data for key in required_keys):
                missing_keys = [key for key in required_keys if key not in data]
                raise ValueError(f"Missing required keys in input data: {missing_keys}")
            
            # Extract data arrays
            close = data['close']
            high = data['high']
            low = data['low']
            
            # Calculate moving averages
            fast_ma = self.calculate_moving_average(close, self.fast_period)
            slow_ma = self.calculate_moving_average(close, self.slow_period)
            
            # Calculate ATR
            atr = self.calculate_atr(high, low, close)
            
            # Initialize arrays for signals and levels
            signal = np.zeros_like(close)
            sl = np.zeros_like(close)
            tp = np.zeros_like(close)
            
            # Generate signals based on MA crossover
            # Long open (1): Fast MA crosses above Slow MA
            long_open = (fast_ma > slow_ma) & (np.roll(fast_ma, 1) <= np.roll(slow_ma, 1))
            signal[long_open] = 1
            
            # Long close (-1): Fast MA crosses below Slow MA
            long_close = (fast_ma < slow_ma) & (np.roll(fast_ma, 1) >= np.roll(slow_ma, 1))
            signal[long_close] = -1
            
            # Short open (2): Fast MA crosses below Slow MA
            short_open = (fast_ma < slow_ma) & (np.roll(fast_ma, 1) >= np.roll(slow_ma, 1))
            signal[short_open] = 2
            
            # Short close (-2): Fast MA crosses above Slow MA
            short_close = (fast_ma > slow_ma) & (np.roll(fast_ma, 1) <= np.roll(slow_ma, 1))
            signal[short_close] = -2
            
            # Calculate SL and TP levels
            # For long positions
            long_mask = (signal == 1)
            sl[long_mask] = close[long_mask] - (atr[long_mask] * self.sl_atr_multiplier)
            tp[long_mask] = close[long_mask] + (atr[long_mask] * self.tp_atr_multiplier)
            
            # For short positions
            short_mask = (signal == 2)
            sl[short_mask] = close[short_mask] + (atr[short_mask] * self.sl_atr_multiplier)
            tp[short_mask] = close[short_mask] - (atr[short_mask] * self.tp_atr_multiplier)
            
            # Log signal statistics
            long_open_count = np.sum(signal == 1)
            long_close_count = np.sum(signal == -1)
            short_open_count = np.sum(signal == 2)
            short_close_count = np.sum(signal == -2)
            logger.info(f"Generated signals: {long_open_count} long opens, {long_close_count} long closes, "
                       f"{short_open_count} short opens, {short_close_count} short closes")
            
            return {
                'time': data['time'],
                'close': close,
                'signal': signal,
                'sl': sl,
                'tp': tp
            }
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            raise 