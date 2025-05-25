import pandas as pd
import numpy as np

class RSIStrategy:
    """
    RSI (Relative Strength Index) Strategy for trading.
    Generates signals based on RSI overbought/oversold conditions with stop loss and take profit levels.
    """
    
    def __init__(self, rsi_period=14, overbought=70, oversold=30, sl_atr_multiplier=2, tp_atr_multiplier=3, atr_period=14):
        """
        Initialize RSI Strategy parameters.
        
        Args:
            rsi_period (int): Period for RSI calculation
            overbought (int): RSI level considered overbought
            oversold (int): RSI level considered oversold
            sl_atr_multiplier (float): Multiplier for ATR to set stop loss
            tp_atr_multiplier (float): Multiplier for ATR to set take profit
            atr_period (int): Period for ATR calculation
        """
        # Convert parameters to appropriate types
        self.rsi_period = int(rsi_period)
        self.overbought = float(overbought)
        self.oversold = float(oversold)
        self.sl_atr_multiplier = float(sl_atr_multiplier)
        self.tp_atr_multiplier = float(tp_atr_multiplier)
        self.atr_period = int(atr_period)
        
        # Validate parameters
        if self.rsi_period <= 0:
            raise ValueError("RSI period must be greater than 0")
        if self.atr_period <= 0:
            raise ValueError("ATR period must be greater than 0")
        if self.overbought <= self.oversold:
            raise ValueError("Overbought level must be greater than oversold level")
        if self.sl_atr_multiplier <= 0:
            raise ValueError("Stop loss ATR multiplier must be greater than 0")
        if self.tp_atr_multiplier <= 0:
            raise ValueError("Take profit ATR multiplier must be greater than 0")

    def calculate_rsi(self, data):
        """Calculate RSI using custom implementation"""
        # Calculate price changes
        delta = data['close'].diff()
        
        # Separate gains and losses
        gain = (delta.where(delta > 0, 0)).rolling(window=int(self.rsi_period)).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=int(self.rsi_period)).mean()
        
        # Calculate RS and RSI
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    def calculate_atr(self, data):
        """Calculate ATR using custom implementation"""
        # Calculate True Range
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        # True Range is the maximum of the three
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        # Calculate ATR
        atr = true_range.rolling(window=int(self.atr_period)).mean()
        
        return atr

    def calculate_sma(self, data, period):
        """Calculate Simple Moving Average"""
        return data['close'].rolling(window=period).mean()

    def generate_signals(self, data):
        """
        Generate trading signals based on RSI and ATR.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with signals and SL/TP levels
        """
        # Create a copy of the input data
        df = data.copy()
        
        # Calculate indicators
        df['RSI'] = self.calculate_rsi(df)
        df['ATR'] = self.calculate_atr(df)
        df['SMA20'] = self.calculate_sma(df, 20)
        df['SMA50'] = self.calculate_sma(df, 50)
        
        # Initialize signal column
        df['Signal'] = None
        
        # Generate signals based on RSI levels and trend confirmation
        # Buy conditions:
        # 1. RSI crosses below oversold level
        # 2. Price is above SMA20 (short-term uptrend)
        # 3. SMA20 is above SMA50 (long-term uptrend)
        buy_condition = (
            (df['RSI'] < self.oversold) & 
            (df['close'] > df['SMA20']) & 
            (df['SMA20'] > df['SMA50'])
        )
        
        # Sell conditions:
        # 1. RSI crosses above overbought level
        # 2. Price is below SMA20 (short-term downtrend)
        # 3. SMA20 is below SMA50 (long-term downtrend)
        sell_condition = (
            (df['RSI'] > self.overbought) & 
            (df['close'] < df['SMA20']) & 
            (df['SMA20'] < df['SMA50'])
        )
        
        # Apply signals
        df.loc[buy_condition, 'Signal'] = 'Buy'
        df.loc[sell_condition, 'Signal'] = 'Sell'
        
        # Calculate Stop Loss and Take Profit levels
        df['SL'] = None
        df['TP'] = None
        
        # For Buy signals
        buy_mask = df['Signal'] == 'Buy'
        df.loc[buy_mask, 'SL'] = df.loc[buy_mask, 'close'] - (df.loc[buy_mask, 'ATR'] * self.sl_atr_multiplier)
        df.loc[buy_mask, 'TP'] = df.loc[buy_mask, 'close'] + (df.loc[buy_mask, 'ATR'] * self.tp_atr_multiplier)
        
        # For Sell signals
        sell_mask = df['Signal'] == 'Sell'
        df.loc[sell_mask, 'SL'] = df.loc[sell_mask, 'close'] + (df.loc[sell_mask, 'ATR'] * self.sl_atr_multiplier)
        df.loc[sell_mask, 'TP'] = df.loc[sell_mask, 'close'] - (df.loc[sell_mask, 'ATR'] * self.tp_atr_multiplier)
        
        # Clean up the DataFrame
        result_df = df[['time', 'open', 'high', 'low', 'close', 'volume', 'Signal', 'SL', 'TP']].copy()
        
        # Remove any rows with NaN values
        result_df = result_df.dropna()
        
        return result_df 