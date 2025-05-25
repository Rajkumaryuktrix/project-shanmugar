import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import numba
from numba import jit

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class TradeMetrics:
    """Data class to store trade metrics"""
    trade_id: int
    open_action: str
    close_action: Optional[str]
    trade_type: str
    open_time: pd.Timestamp
    close_time: Optional[pd.Timestamp]
    volume: int
    open_price: float
    close_price: Optional[float]
    pnl: Optional[float]
    mae: float
    mfe: float
    achieved_rr: float
    opportunity_rr: float
    duration: Optional[pd.Timedelta]
    capital_invested: float
    capital_risked_pct: float
    roi: float
    account_balance: float
    commission: float

class BacktestingTradeEngine:
    def __init__(self, separate_close_signals=False, parallel_opening=False):
        # Pre-allocate DataFrames with fixed size for better memory management
        self.signal_trades_columns = [
            'Trade_id', 'Trade_Open_Action', 'Trade_Close_Action', 'Trade_Type',
            'Open_time', 'Close_time', 'volume', 'Open_price', 'Close_Price', 'PnL',
            'MAE', 'MFE', 'Achieved_RR', 'Opportunity_RR', 'Duration',
            'Capital_Invested', 'Capital_Risked%', 'ROI', 'Account_Balance', 'Commission'
        ]
        
        self.sltp_trades_columns = [
            'Trade_id', 'Trade_Open_Action', 'Trade_Type', 'Open_time', 'Close_time',
            'volume', 'Open_price', 'SL', 'TP', 'Close_Price', 'PnL', 'Trade_Result',
            'MAE', 'MFE', 'Achieved_RR', 'Opportunity_RR', 'Duration',
            'Capital_Invested', 'Capital_Risked%', 'ROI', 'Account_Balance', 'Commission'
        ]
        
        # Initialize with empty DataFrames
        self.SignalTrades = pd.DataFrame(columns=self.signal_trades_columns)
        self.SLTPTrades = pd.DataFrame(columns=self.sltp_trades_columns)
        self.SeparateCloseSignals = separate_close_signals
        self.ParallelOpening = parallel_opening
        self.AcBalance = 0

    @staticmethod
    @jit(nopython=True)
    def _calculate_invested_amount(lot_size: float, asset_price: float, leverage: float) -> float:
        """Calculate invested amount using Numba-accelerated computation"""
        if leverage <= 0:
            return 0.0
        return (asset_price * lot_size) / (leverage / 100)

    @staticmethod
    @jit(nopython=True)
    def _calculate_trade_metrics(open_price: float, close_price: float, volume: int, 
                               mae: float, mfe: float, capital_invested: float, 
                               account_balance: float, commission: float) -> Tuple[float, float, float, float, float]:
        """Calculate trade metrics using Numba-accelerated computation"""
        pnl = (close_price - open_price) * volume - commission
        achieved_rr = 0.0 if mae == 0 else (pnl/volume) / abs(mae)
        opportunity_rr = 0.0 if mae == 0 else mfe / abs(mae)
        capital_risked_pct = (capital_invested/account_balance) * 100
        roi = (pnl/capital_invested) * 100
        return pnl, achieved_rr, opportunity_rr, capital_risked_pct, roi

    def signal_based_trade_executor(self, data: pd.DataFrame, balance: float, leverage: float,
                                  volume: int, commission: float) -> Dict:
        """
        Execute trades based on signals.
        
        Args:
            data (pd.DataFrame): DataFrame with columns ['time', 'Price', 'Signal']
            balance (float): Initial account balance
            leverage (float): Account leverage
            volume (int): Trading volume
            commission (float): Commission per trade
            
        Returns:
            Dict: Dictionary containing trades and metrics
        """
        try:
            trades = []
            current_position = None
            entry_price = 0
            entry_time = None
            
            for idx, row in data.iterrows():
                if current_position is None:  # No position
                    if row['Signal'] == 'Buy':
                        current_position = 'Long'
                        entry_price = row['Price']
                        entry_time = row['time']
                    elif row['Signal'] == 'Sell':
                        current_position = 'Short'
                        entry_price = row['Price']
                        entry_time = row['time']
                else:  # Have position
                    if (current_position == 'Long' and row['Signal'] == 'Sell') or \
                       (current_position == 'Short' and row['Signal'] == 'Buy'):
                        # Calculate PnL
                        if current_position == 'Long':
                            pnl = (row['Price'] - entry_price) * volume
                        else:  # Short
                            pnl = (entry_price - row['Price']) * volume
                        
                        # Apply commission
                        pnl -= commission
                        
                        # Update balance
                        balance += pnl
                        
                        # Record trade
                        trades.append({
                            'time': entry_time,
                            'Price': entry_price,
                            'Signal': 'Buy' if current_position == 'Long' else 'Sell',
                            'Exit_Time': row['time'],
                            'Exit_Price': row['Price'],
                            'PnL': pnl,
                            'Balance': balance
                        })
                        
                        # Reset position
                        current_position = None
                        entry_price = 0
                        entry_time = None
            
            # Calculate metrics
            metrics = self._calculate_performance_metrics(trades)
            
            return {
                'trades': trades,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Error in signal-based trade execution: {str(e)}")
            raise

    def sltp_based_trade_executor(self, data: pd.DataFrame, balance: float, leverage: float,
                                volume: int, commission: float) -> Dict:
        """
        Execute trades based on stop-loss and take-profit levels.
        
        Args:
            data (pd.DataFrame): DataFrame with columns ['time', 'Price', 'Signal', 'SL', 'TP']
            balance (float): Initial account balance
            leverage (float): Account leverage
            volume (int): Trading volume
            commission (float): Commission per trade
            
        Returns:
            Dict: Dictionary containing trades and metrics
        """
        try:
            # Validate and convert numeric columns
            required_columns = ['time', 'Price', 'Signal', 'SL', 'TP']
            for col in required_columns:
                if col not in data.columns:
                    raise ValueError(f"Required column '{col}' not found in data")
            
            # Convert numeric columns to float
            numeric_columns = ['Price', 'SL', 'TP']
            for col in numeric_columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Drop rows with NaN values in numeric columns
            data = data.dropna(subset=numeric_columns)
            
            trades = []
            current_position = None
            entry_price = 0.0
            entry_time = None
            stop_loss = 0.0
            take_profit = 0.0
            
            for idx, row in data.iterrows():
                if current_position is None:  # No position
                    if row['Signal'] == 'Buy':
                        current_position = 'Long'
                        entry_price = float(row['Price'])
                        entry_time = row['time']
                        stop_loss = float(row['SL'])
                        take_profit = float(row['TP'])
                    elif row['Signal'] == 'Sell':
                        current_position = 'Short'
                        entry_price = float(row['Price'])
                        entry_time = row['time']
                        stop_loss = float(row['SL'])
                        take_profit = float(row['TP'])
                else:  # Have position
                    current_price = float(row['Price'])
                    # Check for stop-loss or take-profit
                    if current_position == 'Long':
                        if current_price <= stop_loss or current_price >= take_profit:
                            # Calculate PnL
                            pnl = (current_price - entry_price) * volume
                            pnl -= commission
                            balance += pnl
                            
                            # Record trade
                            trades.append({
                                'time': entry_time,
                                'Price': entry_price,
                                'Signal': 'Buy',
                                'Exit_Time': row['time'],
                                'Exit_Price': current_price,
                                'PnL': pnl,
                                'Balance': balance
                            })
                            
                            # Reset position
                            current_position = None
                            entry_price = 0.0
                            entry_time = None
                    else:  # Short position
                        if current_price >= stop_loss or current_price <= take_profit:
                            # Calculate PnL
                            pnl = (entry_price - current_price) * volume
                            pnl -= commission
                            balance += pnl
                            
                            # Record trade
                            trades.append({
                                'time': entry_time,
                                'Price': entry_price,
                                'Signal': 'Sell',
                                'Exit_Time': row['time'],
                                'Exit_Price': current_price,
                                'PnL': pnl,
                                'Balance': balance
                            })
                            
                            # Reset position
                            current_position = None
                            entry_price = 0.0
                            entry_time = None
            
            # Calculate metrics
            metrics = self._calculate_performance_metrics(trades)
        
            return {
                'trades': trades,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Error in SLTP-based trade execution: {str(e)}")
            raise

    @staticmethod
    def _calculate_performance_metrics(trades: List[Dict]) -> Dict:
        """Calculate performance metrics using vectorized operations"""
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'average_pnl': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }
        
        # Convert to numpy arrays for faster calculations
        pnls = np.array([trade['PnL'] for trade in trades])
        
        # Calculate durations if available
        if 'Exit_Time' in trades[0] and 'time' in trades[0]:
            durations = np.array([
                (pd.to_datetime(trade['Exit_Time']) - pd.to_datetime(trade['time'])).total_seconds()
                for trade in trades
            ])
        else:
            durations = np.array([0])
        
        winning_trades = pnls > 0
        losing_trades = pnls < 0
        
        return {
            'total_trades': len(trades),
            'winning_trades': np.sum(winning_trades),
            'losing_trades': np.sum(losing_trades),
            'win_rate': np.mean(winning_trades) * 100,
            'total_pnl': np.sum(pnls),
            'average_pnl': np.mean(pnls),
            'max_drawdown': np.min(pnls),
            'sharpe_ratio': np.mean(pnls) / np.std(pnls) if len(pnls) > 1 else 0,
            'average_duration': np.mean(durations),
            'profit_factor': abs(np.sum(pnls[winning_trades]) / np.sum(pnls[losing_trades])) 
                            if np.sum(pnls[losing_trades]) != 0 else float('inf')
        }