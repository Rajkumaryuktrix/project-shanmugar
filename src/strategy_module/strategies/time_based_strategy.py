import pandas as pd
import numpy as np
import logging
from datetime import datetime, time
from typing import Dict, List, Optional, Union

# Setup logging
logger = logging.getLogger(__name__)

class TimeBasedStrategy:
    """
    Time-based trading strategy that opens and closes positions at specified times
    with customizable stop-loss and take-profit levels.
    
    Features:
    - Multiple entry/exit times support
    - Percentage-based SL/TP levels
    - Flexible time format support
    - Position direction control (long/short)
    """
    
    def __init__(self, 
                 entry_times: List[str],
                 exit_times: List[str],
                 sl_percentage: float = 1.0,
                 tp_percentage: float = 2.0,
                 position_direction: str = 'long'):
        """
        Initialize Time-based Strategy parameters.
        
        Args:
            entry_times (List[str]): List of entry times in 'HH:MM' format
            exit_times (List[str]): List of exit times in 'HH:MM' format
            sl_percentage (float): Stop loss percentage from entry price
            tp_percentage (float): Take profit percentage from entry price
            position_direction (str): 'long' or 'short' position direction
        """
        self.entry_times = self._validate_and_convert_times(entry_times)
        self.exit_times = self._validate_and_convert_times(exit_times)
        self.sl_percentage = float(sl_percentage)
        self.tp_percentage = float(tp_percentage)
        self.position_direction = position_direction.lower()
        
        # Validate parameters
        self._validate_parameters()
        
        logger.info(f"Time-based Strategy initialized with parameters:")
        logger.info(f"Entry times: {[t.strftime('%H:%M') for t in self.entry_times]}")
        logger.info(f"Exit times: {[t.strftime('%H:%M') for t in self.exit_times]}")
        logger.info(f"SL: {self.sl_percentage}%, TP: {self.tp_percentage}%")
        logger.info(f"Position direction: {self.position_direction}")

    def _validate_and_convert_times(self, time_list: List[str]) -> List[time]:
        """Convert and validate time strings to time objects"""
        converted_times = []
        for t in time_list:
            try:
                # Try different time formats
                for fmt in ['%H:%M', '%H:%M:%S', '%I:%M %p', '%I:%M:%S %p']:
                    try:
                        converted_times.append(datetime.strptime(t, fmt).time())
                        break
                    except ValueError:
                        continue
                else:
                    raise ValueError(f"Invalid time format: {t}")
            except Exception as e:
                logger.error(f"Error converting time {t}: {str(e)}")
                raise
        return sorted(converted_times)

    def _validate_parameters(self):
        """Validate strategy parameters"""
        if not self.entry_times:
            raise ValueError("At least one entry time must be specified")
        if not self.exit_times:
            raise ValueError("At least one exit time must be specified")
        if self.sl_percentage <= 0:
            raise ValueError("Stop loss percentage must be greater than 0")
        if self.tp_percentage <= 0:
            raise ValueError("Take profit percentage must be greater than 0")
        if self.position_direction not in ['long', 'short']:
            raise ValueError("Position direction must be either 'long' or 'short'")
        
        # Validate time sequence
        if min(self.exit_times) <= max(self.entry_times):
            logger.warning("Some exit times are before or equal to entry times")

    def _is_time_in_list(self, current_time: time, time_list: List[time]) -> bool:
        """Check if current time matches any time in the list"""
        return any(abs((current_time.hour * 60 + current_time.minute) - 
                      (t.hour * 60 + t.minute)) < 1 for t in time_list)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on specified times.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data and 'time' column
            
        Returns:
            pd.DataFrame: DataFrame with signals and SL/TP levels
        """
        try:
            # Validate input data
            required_columns = ['time', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                missing_cols = [col for col in required_columns if col not in data.columns]
                raise ValueError(f"Missing required columns in input data: {missing_cols}")
            
            # Create a copy of the input data
            df = data.copy()
            
            # Convert time column to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(df['time']):
                df['time'] = pd.to_datetime(df['time'])
            
            # Initialize signal columns
            df['Signal'] = None
            df['SL'] = None
            df['TP'] = None
            
            # Generate signals based on time
            for idx, row in df.iterrows():
                current_time = row['time'].time()
                
                # Check for entry signal
                if self._is_time_in_list(current_time, self.entry_times):
                    df.at[idx, 'Signal'] = 'Buy' if self.position_direction == 'long' else 'Sell'
                    current_price = row['close']
                    
                    # Calculate SL and TP levels
                    if self.position_direction == 'long':
                        df.at[idx, 'SL'] = current_price * (1 - self.sl_percentage / 100)
                        df.at[idx, 'TP'] = current_price * (1 + self.tp_percentage / 100)
                    else:  # short position
                        df.at[idx, 'SL'] = current_price * (1 + self.sl_percentage / 100)
                        df.at[idx, 'TP'] = current_price * (1 - self.tp_percentage / 100)
                
                # Check for exit signal
                elif self._is_time_in_list(current_time, self.exit_times):
                    df.at[idx, 'Signal'] = 'Sell' if self.position_direction == 'long' else 'Buy'
            
            # Clean up the DataFrame
            result_df = df[['time', 'close', 'Signal', 'SL', 'TP']].copy()
            result_df = result_df.rename(columns={'close': 'Price'})
            
            # Remove any rows with NaN values
            result_df = result_df.dropna()
            
            # Validate if any signals were generated
            if result_df['Signal'].isna().all():
                logger.warning("No trading signals were generated. Check time parameters and data.")
            
            # Log signal statistics
            buy_signals = (result_df['Signal'] == 'Buy').sum()
            sell_signals = (result_df['Signal'] == 'Sell').sum()
            logger.info(f"Generated {buy_signals} buy signals and {sell_signals} sell signals")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            raise 