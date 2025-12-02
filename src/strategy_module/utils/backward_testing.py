import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
import numba
from numba import jit
import random
from datetime import datetime
from abc import ABC, abstractmethod

# Import from the same directory
from .brokerage_tax_calculator import IndianTradeCostCalculator

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler with formatting
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

@dataclass
class CandleData:
    """Data class to store candle data in numpy arrays for efficient processing"""
    timestamp: np.ndarray  # Unix timestamps
    open: np.ndarray      # Open prices
    high: np.ndarray      # High prices
    low: np.ndarray       # Low prices
    close: np.ndarray     # Close prices
    volume: np.ndarray    # Volume
    oi: np.ndarray        # Open Interest
    signal: np.ndarray    # Trading signals
    sl: Optional[np.ndarray] = None  # Stop loss levels
    tp: Optional[np.ndarray] = None  # Take profit levels

    @classmethod
    def from_list(cls, candles: List[List], signals: Optional[List] = None, 
                 sl_levels: Optional[List] = None, tp_levels: Optional[List] = None):
        """Create CandleData from list of candles"""
        if not candles:
            raise ValueError("Empty candle data")
            
        # Convert to numpy arrays
        data = np.array(candles, dtype=object)
        
        # Validate data types and missing values
        for i, row in enumerate(data):
            try:
                # Validate timestamp format
                datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
                
                # Validate numeric values
                for j in range(1, 7):  # OHLCV and OI columns
                    if pd.isna(row[j]) or not isinstance(float(row[j]), (int, float)):
                        raise ValueError(f"Invalid numeric value at row {i}, column {j}")
            except ValueError as e:
                raise ValueError(f"Data validation failed at row {i}: {str(e)}")
        
        # Extract timestamps and convert to unix timestamps
        timestamps = np.array([int(datetime.strptime(ts, '%Y-%m-%d %H:%M:%S').timestamp()) 
                             for ts in data[:, 0]])
        
        # Convert price and volume data to float arrays
        open_prices = data[:, 1].astype(float)
        high_prices = data[:, 2].astype(float)
        low_prices = data[:, 3].astype(float)
        close_prices = data[:, 4].astype(float)
        volumes = data[:, 5].astype(float)
        oi_values = data[:, 6].astype(float)
        
        # Convert signals and levels if provided
        signal_array = np.array(signals, dtype=float) if signals is not None else np.zeros(len(candles))
        sl_array = np.array(sl_levels, dtype=float) if sl_levels is not None else None
        tp_array = np.array(tp_levels, dtype=float) if tp_levels is not None else None
        
        return cls(
            timestamp=timestamps,
            open=open_prices,
            high=high_prices,
            low=low_prices,
            close=close_prices,
            volume=volumes,
            oi=oi_values,
            signal=signal_array,
            sl=sl_array,
            tp=tp_array
        )

class BacktestingEngine:
    """Unified backtesting engine that handles both signal-based and SLTP-based exits"""
    
    def __init__(self, candle_data: CandleData, **kwargs):
        try:
            logger.info("Initializing BacktestingEngine")
            self.data = candle_data
            
            # Validate numeric parameters
            self.initial_balance = float(kwargs.get('initial_balance', 1000.0))
            if self.initial_balance <= 0:
                raise ValueError("initial_balance must be positive")
                
            self.volume = float(kwargs.get('volume', 10.0))
            if self.volume <= 0:
                raise ValueError("volume must be positive")
                
            self.max_position_size = float(kwargs.get('max_position_size', float('inf')))
            if self.max_position_size <= 0:
                raise ValueError("max_position_size must be positive")
                
            self.risk_per_trade = float(kwargs.get('risk_per_trade', 0.02))
            if not 0 < self.risk_per_trade <= 1:
                raise ValueError("risk_per_trade must be between 0 and 1")
                
            self.leverage = float(kwargs.get('leverage', 1.0))
            if self.leverage <= 0:
                raise ValueError("leverage must be positive")
                
            self.slippage_value = float(kwargs.get('slippage_value', 0.0))
            if self.slippage_value < 0:
                raise ValueError("slippage_value must be non-negative")
                
            self.min_position_size_ratio = float(kwargs.get('min_position_size_ratio', 0.3))
            if not 0 < self.min_position_size_ratio <= 1:
                raise ValueError("min_position_size_ratio must be between 0 and 1")
                
            self.min_fill_ratio = float(kwargs.get('min_fill_ratio', 0.3))
            if not 0 < self.min_fill_ratio <= 1:
                raise ValueError("min_fill_ratio must be between 0 and 1")
            
            # Validate string parameters
            self.slippage_model = str(kwargs.get('slippage_model', 'none'))
            valid_slippage_models = ['none', 'fixed', 'percentage', 'random']
            if self.slippage_model not in valid_slippage_models:
                raise ValueError(f"slippage_model must be one of {valid_slippage_models}")
                
            self.position_sizing_method = str(kwargs.get('position_sizing_method', 'fixed'))
            valid_sizing_methods = ['fixed', 'risk_based']
            if self.position_sizing_method not in valid_sizing_methods:
                raise ValueError(f"position_sizing_method must be one of {valid_sizing_methods}")
                
            self.segment = str(kwargs.get('segment', 'intraday'))
            valid_segments = ['intraday', 'delivery']
            if self.segment not in valid_segments:
                raise ValueError(f"segment must be one of {valid_segments}")
            
            # Validate integer parameters
            self.max_open_positions = int(kwargs.get('max_open_positions', 1))
            if self.max_open_positions <= 0:
                raise ValueError("max_open_positions must be positive")
            
            # Validate boolean parameters
            self.allow_partial_fills = bool(kwargs.get('allow_partial_fills', False))
            self.parallel_opening = bool(kwargs.get('parallel_opening', False))
            self.use_sltp = bool(kwargs.get('use_sltp', True))
            self.force_signal_exits = bool(kwargs.get('force_signal_exits', False))
            self.strict_sltp = bool(kwargs.get('strict_sltp', False))

            # SL/TP validation
            if self.use_sltp:
                if self.data.sl is None or self.data.tp is None:
                    msg = ("SL/TP levels are missing in CandleData but use_sltp=True. "
                           "Backtesting will fall back to signal-based exits only.")
                    if self.strict_sltp:
                        logger.error(msg)
                        raise ValueError("SL/TP required but missing in CandleData when use_sltp=True and strict_sltp=True.")
                    else:
                        logger.warning(msg)

            self.tax_calculator = IndianTradeCostCalculator()
            self.trades = []
            self.balance_history = np.array([self.initial_balance])
            self.equity_curve = np.array([self.initial_balance])
            self.drawdown_curve = np.array([0.0])
            
            # Log engine configuration
            engine_type = "SLTP-based" if self.use_sltp else "Signal-based"
            logger.info(f"Engine configured as {engine_type} with {'parallel' if self.parallel_opening else 'single'} position mode")
        except Exception as e:
            logger.error(f"Error during BacktestingEngine initialization: {e}", exc_info=True)
            raise

    @staticmethod
    @jit(nopython=True)
    def _calculate_pnl_vectorized(entry_prices: np.ndarray, exit_prices: np.ndarray, 
                                volumes: np.ndarray, position_types: np.ndarray, 
                                commissions: np.ndarray) -> np.ndarray:
        position_multipliers = np.where(position_types == 1, 1.0, -1.0)
        price_diffs = exit_prices - entry_prices
        return (price_diffs * position_multipliers * volumes) - commissions

    def calculate_position_size(self, price: float, volume: float) -> float:
        """Calculate position size based on risk management rules."""
        if price <= 0 or volume <= 0:
            return 0.0
        
        min_position_size = volume * self.min_position_size_ratio
        
        if self.position_sizing_method == 'fixed':
            # For fixed sizing, use the requested volume, capped by max_position_size if not infinite
            if self.max_position_size != float('inf'):
                position_size = min(volume, self.max_position_size)
            else:
                position_size = volume
        else:  # risk-based sizing
            max_position_value = self.initial_balance * self.risk_per_trade
            max_position_quantity = max_position_value / price
            if self.max_position_size != float('inf'):
                max_position_quantity = min(max_position_quantity, self.max_position_size)
            position_size = min(max_position_quantity, volume)
        
        # For partial fills, ensure minimum position size is respected
        if self.allow_partial_fills:
            if position_size < min_position_size:
                position_size = min_position_size
        else:
            if position_size < min_position_size:
                return 0.0
            if self.position_sizing_method == 'fixed':
                position_size = min(position_size, volume)
        return position_size

    def apply_slippage(self, price: float, volume: float, direction: str) -> float:
        """Apply slippage to price based on model and direction."""
        try:
            if self.slippage_model == 'none':
                return price
                
            if self.slippage_model == 'fixed':
                slippage = self.slippage_value
            elif self.slippage_model == 'percentage':
                slippage = price * (self.slippage_value / 100)
            elif self.slippage_model == 'random':
                slippage = price * (np.random.uniform(0, self.slippage_value) / 100)
            else:
                raise ValueError(f"Invalid slippage model: {self.slippage_model}")
            
            # Apply slippage based on direction
            if direction == 'long':
                return price * (1 + slippage/price)  # Increase price for long entries
            elif direction == 'short':
                return price * (1 - slippage/price)  # Decrease price for short entries
            else:
                raise ValueError(f"Invalid direction: {direction}")
                
        except Exception as e:
            logger.error(f"Error applying slippage: {str(e)}")
            raise

    def run_backtest(self) -> Dict:
        try:
            start_time = datetime.now()
            n_candles = len(self.data.timestamp)
            current_balance = self.initial_balance
            open_positions = []
            trade_id_counter = 0
            balance_history = np.zeros(n_candles + 1)
            equity_curve = np.zeros(n_candles + 1)
            drawdown_curve = np.zeros(n_candles + 1)
            balance_history[0] = current_balance
            equity_curve[0] = current_balance
            long_open_signals = np.where(self.data.signal == 1, 1, 0)
            long_close_signals = np.where(self.data.signal == -1, 1, 0)
            short_open_signals = np.where(self.data.signal == 2, 1, 0)
            short_close_signals = np.where(self.data.signal == -2, 1, 0)
            entry_signals = np.where(long_open_signals == 1, 1, 
                                   np.where(short_open_signals == 1, -1, 0))
            exit_signals = np.where(long_close_signals == 1, 1,
                                  np.where(short_close_signals == 1, 1, 0))

            # Calculate minimum required capital for smallest possible trade
            min_volume = self.volume * self.min_position_size_ratio
            min_position_value = min_volume * self.data.close[0]  # Use first price as reference
            min_required_capital = min_position_value / self.leverage

            # If initial balance is already insufficient, log and return early
            if current_balance < min_required_capital:
                logger.info(f"Initial balance {current_balance:.2f} is insufficient for minimum trade size. Minimum required: {min_required_capital:.2f}")
                return {
                    'trades': [],
                    'balance_history': [current_balance],
                    'equity_curve': [current_balance],
                    'drawdown_curve': [0.0]
                }

            for i in range(n_candles):
                current_time = self.data.timestamp[i]
                current_price = self.data.close[i]
                current_signal = entry_signals[i]
                
                # Update balance history at the start of each candle
                balance_history[i + 1] = current_balance
                
                # Calculate equity including unrealized PnL
                equity = current_balance
                if open_positions:
                    position_data = np.array([(p['Entry_Price'], p['Position_Size'], 
                                            1 if p['Trade_Type'] == 'long' else -1,
                                            p['Initial_Commission']) 
                                           for p in open_positions], dtype=np.float64)
                    unrealized_pnl = self._calculate_pnl_vectorized(
                        position_data[:, 0],
                        np.full(len(position_data), current_price, dtype=np.float64),
                        position_data[:, 1],
                        position_data[:, 2],
                        position_data[:, 3]
                    )
                    equity += np.sum(unrealized_pnl)
                equity_curve[i + 1] = equity
                
                # Calculate drawdown
                peak_balance = np.max(equity_curve[:i + 2])
                drawdown = (peak_balance - equity) / peak_balance if peak_balance > 0 else 0
                drawdown_curve[i + 1] = drawdown

                # Check if current balance is sufficient for minimum trade
                min_position_value = min_volume * current_price
                min_required_capital = min_position_value / self.leverage
                if current_balance < min_required_capital:
                    logger.info(f"Balance {current_balance:.2f} fell below minimum required {min_required_capital:.2f} at index {i}. Stopping backtest.")
                    # Close any open positions
                    for position in open_positions:
                        exit_price = self.apply_slippage(current_price, position['Position_Size'], 
                                                      'short' if position['Trade_Type'] == 'long' else 'long')
                        pnl = self._calculate_pnl_vectorized(
                            np.array([position['Entry_Price']]),
                            np.array([exit_price]),
                            np.array([position['Position_Size']]),
                            np.array([1 if position['Trade_Type'] == 'long' else -1]),
                            np.array([position['Initial_Commission']])
                        )[0]
                        current_balance += pnl
                        position['Exit_Price'] = exit_price
                        position['Exit_Time'] = current_time
                        position['PnL'] = pnl
                        position['Balance'] = current_balance
                        position['Trade_Close_Action'] = 'insufficient_balance'
                        self.trades.append(position)
                    open_positions = []
                    # Trim arrays to current length
                    balance_history = balance_history[:i + 2]
                    equity_curve = equity_curve[:i + 2]
                    drawdown_curve = drawdown_curve[:i + 2]
                    break

                positions_to_close = []
                if not self.force_signal_exits or exit_signals[i]:
                    for position in open_positions:
                        should_close = False
                        close_reason = None
                        if (position['Trade_Type'] == 'long' and long_close_signals[i] == 1) or \
                           (position['Trade_Type'] == 'short' and short_close_signals[i] == 1):
                            should_close = True
                            close_reason = 'signal_close'
                        elif self.use_sltp and not should_close:
                            if position['Trade_Type'] == 'long':
                                if position['SL'] is not None and current_price <= position['SL']:
                                    should_close = True
                                    close_reason = 'stop_loss'
                                elif position['TP'] is not None and current_price >= position['TP']:
                                    should_close = True
                                    close_reason = 'take_profit'
                            else:
                                if position['SL'] is not None and current_price >= position['SL']:
                                    should_close = True
                                    close_reason = 'stop_loss'
                                elif position['TP'] is not None and current_price <= position['TP']:
                                    should_close = True
                                    close_reason = 'take_profit'
                        if should_close:
                            pnl = self._calculate_pnl_vectorized(
                                np.array([position['Entry_Price']], dtype=np.float64),
                                np.array([current_price], dtype=np.float64),
                                np.array([position['Position_Size']], dtype=np.float64),
                                np.array([1 if position['Trade_Type'] == 'long' else -1], dtype=np.float64),
                                np.array([position['Initial_Commission']], dtype=np.float64)
                            )[0]
                            trade_record = {
                                'Entry_Time': position['Entry_Time'],
                                'Exit_Time': current_time,
                                'Entry_Price': position['Entry_Price'],
                                'Exit_Price': current_price,
                                'Position_Size': position['Position_Size'],
                                'Trade_Type': position['Trade_Type'],
                                'Trade_Close_Action': close_reason,
                                'Trade_ID': position['Trade_ID'],
                                'PnL': pnl,
                                'Commission': position['Initial_Commission'],
                                'Balance': current_balance + pnl - position['Initial_Commission']
                            }
                            self.trades.append(trade_record)
                            current_balance = trade_record['Balance']
                            positions_to_close.append(position)
                for position in positions_to_close:
                    open_positions.remove(position)
                if entry_signals[i] != 0:
                    # Parallel position enforcement
                    if not self.parallel_opening:
                        if open_positions:
                            # Only one position allowed at a time
                            continue
                    else:
                        if len(open_positions) >= self.max_open_positions:
                            continue
                    position_type = 'long' if entry_signals[i] > 0 else 'short'
                    try:
                        entry_price = self.apply_slippage(current_price, self.volume, position_type)
                    except Exception as e:
                        continue
                    position_size = self.calculate_position_size(entry_price, self.volume)
                    min_position_size = self.volume * self.min_position_size_ratio
                    if position_size >= min_position_size:
                        required_margin = position_size * entry_price / self.leverage
                        if required_margin <= current_balance:
                            trade_id_counter += 1
                            try:
                                trade = self.tax_calculator.Trade(
                                    segment=self.segment,
                                    buy_value=entry_price * position_size,
                                    sell_value=0.0
                                )
                                trade_costs = self.tax_calculator.calc_charges(trade)
                                initial_commission = trade_costs['total_cost']
                            except Exception as e:
                                continue
                            position = {
                                'Entry_Time': current_time,
                                'Entry_Price': entry_price,
                                'Position_Size': position_size,
                                'Trade_Type': position_type,
                                'SL': self.data.sl[i] if self.data.sl is not None else None,
                                'TP': self.data.tp[i] if self.data.tp is not None else None,
                                'Initial_Commission': initial_commission,
                                'Trade_ID': trade_id_counter,
                                'Tax_Breakdown': trade_costs
                            }
                            open_positions.append(position)
                            current_balance -= initial_commission
                        else:
                            logger.warning(f"Insufficient margin for position at index {i} (required: {required_margin}, available: {current_balance})")
                    else:
                        logger.warning(f"Position size {position_size} below minimum {min_position_size} at index {i}")
            if open_positions:
                final_price = self.data.close[-1]
                for position in open_positions:
                    pnl = self._calculate_pnl_vectorized(
                        np.array([position['Entry_Price']], dtype=np.float64),
                        np.array([final_price], dtype=np.float64),
                        np.array([position['Position_Size']], dtype=np.float64),
                        np.array([1 if position['Trade_Type'] == 'long' else -1], dtype=np.float64),
                        np.array([position['Initial_Commission']], dtype=np.float64)
                    )[0]
                    trade_record = {
                        'Entry_Time': position['Entry_Time'],
                        'Exit_Time': self.data.timestamp[-1],
                        'Entry_Price': position['Entry_Price'],
                        'Exit_Price': final_price,
                        'Position_Size': position['Position_Size'],
                        'Trade_Type': position['Trade_Type'],
                        'Trade_Close_Action': 'final_close',
                        'Trade_ID': position['Trade_ID'],
                        'PnL': pnl,
                        'Commission': position['Initial_Commission'],
                        'Balance': current_balance + pnl - position['Initial_Commission']
                    }
                    self.trades.append(trade_record)
                    current_balance = trade_record['Balance']
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            logger.info(f"Backtest completed in {execution_time:.2f} seconds with {len(self.trades)} trades executed")
            
            # Create strategy configuration dictionary
            strategy_config = {
                'initial_balance': self.initial_balance,
                'volume': self.volume,
                'max_position_size': self.max_position_size,
                'risk_per_trade': self.risk_per_trade,
                'leverage': self.leverage,
                'slippage_model': self.slippage_model,
                'slippage_value': self.slippage_value,
                'allow_partial_fills': self.allow_partial_fills,
                'max_open_positions': self.max_open_positions,
                'position_sizing_method': self.position_sizing_method,
                'parallel_opening': self.parallel_opening,
                'segment': self.segment,
                'min_position_size_ratio': self.min_position_size_ratio,
                'min_fill_ratio': self.min_fill_ratio,
                'use_sltp': self.use_sltp,
                'force_signal_exits': self.force_signal_exits,
                'strict_sltp': self.strict_sltp,
                'execution_time': execution_time,
                'total_trades': len(self.trades)
            }
            
            return {
                'trades': self.trades,
                'balance_history': balance_history.tolist(),
                'equity_curve': equity_curve.tolist(),
                'drawdown_curve': drawdown_curve.tolist(),
                'strategy_config': strategy_config
            }
        except Exception as e:
            logger.error(f"Error in run_backtest: {str(e)}", exc_info=True)
            raise

def create_backtesting_engine(candle_data: CandleData, **kwargs) -> BacktestingEngine:
    """
    Factory function to create backtesting engine.
    
    Args:
        candle_data: CandleData object containing price and signal data
        **kwargs: Additional parameters for the engine
        
    Returns:
        Instance of backtesting engine
    """
    return BacktestingEngine(candle_data, **kwargs)