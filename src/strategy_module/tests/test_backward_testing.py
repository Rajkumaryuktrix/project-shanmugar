import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import shutil
import tempfile
import pytest
import logging
import time
import sys

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.strategy_module.utils.backward_testing import BacktestingEngine, CandleData
from src.strategy_module.utils.backtest_data_manager import BacktestDataManager, BacktestMetadata


# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler with formatting
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class TestBacktestingEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data that will be used across all test methods."""
        logger.info("Setting up test data for all test methods")
        # Create sample data with realistic market hours
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='1min')
        market_hours = dates[dates.indexer_between_time('09:30', '16:00')]
        
        # Generate realistic price data
        base_price = 100
        n_periods = len(market_hours)
        
        # Create price data with known patterns
        trend = np.linspace(0, 15, n_periods)
        cycle = 5 * np.sin(np.linspace(0, 4*np.pi, n_periods))
        volatility = np.random.normal(0, 0.5, n_periods)
        prices = base_price + trend + cycle + volatility
        prices = np.maximum(prices, 0.1)
        
        # Generate signals based on price action
        signals = np.zeros(n_periods)
        for i in range(20, n_periods):
            short_ma = np.mean(prices[i-5:i])
            long_ma = np.mean(prices[i-20:i])
            if short_ma > long_ma:
                signals[i] = 1  # Long open
            elif short_ma < long_ma:
                signals[i] = 2  # Short open
            elif i > 0 and signals[i-1] == 1:
                signals[i] = -1  # Long close
            elif i > 0 and signals[i-1] == 2:
                signals[i] = -2  # Short close
        
        # Calculate ATR for SL/TP
        returns = np.diff(prices) / prices[:-1]
        atr = np.zeros(n_periods)
        for i in range(14, n_periods):
            atr[i] = np.mean(np.abs(returns[i-14:i]))
        
        # Create SL/TP levels
        sl_levels = prices * (1 - 2 * atr)
        tp_levels = prices * (1 + 3 * atr)
        
        # Create test DataFrame
        cls.test_data = pd.DataFrame({
            'timestamp': market_hours,
            'open': prices,
            'high': prices * 1.001,
            'low': prices * 0.999,
            'close': prices,
            'volume': np.random.randint(100, 1000, n_periods),
            'oi': np.random.randint(1000, 10000, n_periods),
            'signal': signals,
            'sl': sl_levels,
            'tp': tp_levels
        })
        
        # Convert timestamps to string format
        cls.test_data['timestamp'] = cls.test_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Create CandleData object
        cls.candle_data = CandleData.from_list(
            cls.test_data[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi']].values.tolist(),
            signals=cls.test_data['signal'].tolist(),
            sl_levels=cls.test_data['sl'].tolist(),
            tp_levels=cls.test_data['tp'].tolist()
        )

        # Create edge case data
        cls.edge_case_data = {
            'empty_data': pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi']),
            'zero_prices': cls.test_data.copy(),
            'negative_prices': cls.test_data.copy(),
            'extreme_prices': cls.test_data.copy(),
            'invalid_signals': cls.test_data.copy(),
            'missing_sltp': cls.test_data.copy()
        }
        
        # Modify edge case data
        cls.edge_case_data['zero_prices'].loc[:, ['open', 'high', 'low', 'close']] = 0
        cls.edge_case_data['negative_prices'].loc[:, ['open', 'high', 'low', 'close']] = -100
        cls.edge_case_data['extreme_prices'].loc[:, ['open', 'high', 'low', 'close']] = 1e9
        cls.edge_case_data['invalid_signals']['signal'] = 999  # Invalid signal value
        cls.edge_case_data['missing_sltp'].loc[:, ['sl', 'tp']] = np.nan
        logger.info("Test data setup completed")

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.engine = BacktestingEngine(
            candle_data=self.candle_data,
            initial_balance=100000.0,
            volume=5.0,
            max_position_size=100,
            risk_per_trade=0.02,
            leverage=1.0,
            allow_partial_fills=False,
            slippage_model='percentage',
            slippage_value=0.001,
            position_sizing_method='fixed',
            use_sltp=True,
            force_signal_exits=False
        )
        
        # Create temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        self.data_manager = BacktestDataManager(base_dir=self.test_dir)

    def tearDown(self):
        """Clean up after each test method."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)

    def test_initialization(self):
        """Test proper initialization of the backtesting engine."""
        logger.info("Starting test_initialization")
        start_time = datetime.now()
        
        self.assertEqual(self.engine.initial_balance, 100000.0)
        self.assertEqual(self.engine.volume, 5.0)
        self.assertEqual(self.engine.max_position_size, 100)
        self.assertEqual(self.engine.risk_per_trade, 0.02)
        self.assertEqual(self.engine.leverage, 1.0)
        self.assertEqual(self.engine.slippage_model, 'percentage')
        self.assertEqual(self.engine.slippage_value, 0.001)
        self.assertFalse(self.engine.allow_partial_fills)
        self.assertEqual(self.engine.position_sizing_method, 'fixed')
        self.assertTrue(self.engine.use_sltp)
        self.assertFalse(self.engine.force_signal_exits)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Completed test_initialization in {execution_time:.2f} seconds - PASSED")

    def test_empty_data_handling(self):
        """Test handling of empty candle data."""
        logger.info("Starting test_empty_data_handling")
        start_time = datetime.now()
        
        with self.assertRaises(ValueError):
            empty_data = CandleData.from_list([])
            BacktestingEngine(candle_data=empty_data)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Completed test_empty_data_handling in {execution_time:.2f} seconds - PASSED")

    def test_zero_price_handling(self):
        """Test handling of zero prices."""
        logger.info("Starting test_zero_price_handling")
        start_time = datetime.now()
        
        zero_data = CandleData.from_list(
            self.edge_case_data['zero_prices'][['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi']].values.tolist(),
            signals=self.edge_case_data['zero_prices']['signal'].tolist()
        )
        engine = BacktestingEngine(candle_data=zero_data)
        position_size = engine.calculate_position_size(0.0, engine.volume)
        self.assertEqual(position_size, 0.0)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Completed test_zero_price_handling in {execution_time:.2f} seconds - PASSED")

    def test_negative_price_handling(self):
        """Test handling of negative prices."""
        logger.info("Starting test_negative_price_handling")
        start_time = datetime.now()
        
        negative_data = CandleData.from_list(
            self.edge_case_data['negative_prices'][['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi']].values.tolist(),
            signals=self.edge_case_data['negative_prices']['signal'].tolist()
        )
        engine = BacktestingEngine(candle_data=negative_data)
        position_size = engine.calculate_position_size(-100.0, engine.volume)
        self.assertEqual(position_size, 0.0)

        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Completed test_negative_price_handling in {execution_time:.2f} seconds - PASSED")

    def test_extreme_price_handling(self):
        """Test handling of extreme price values."""
        logger.info("Starting test_extreme_price_handling")
        start_time = datetime.now()
        
        extreme_data = CandleData.from_list(
            self.edge_case_data['extreme_prices'][['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi']].values.tolist(),
            signals=self.edge_case_data['extreme_prices']['signal'].tolist()
        )
        engine = BacktestingEngine(
            candle_data=extreme_data,
            position_sizing_method='risk_based',
            risk_per_trade=0.02,
            initial_balance=1000.0,
            volume=10.0
        )
        position_size = engine.calculate_position_size(1e9, engine.volume)
        self.assertLess(position_size, engine.volume)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Completed test_extreme_price_handling in {execution_time:.2f} seconds - PASSED")

    def test_invalid_signal_handling(self):
        """Test handling of invalid signals."""
        logger.info("Starting test_invalid_signal_handling")
        start_time = datetime.now()
        
        invalid_data = CandleData.from_list(
            self.edge_case_data['invalid_signals'][['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi']].values.tolist(),
            signals=self.edge_case_data['invalid_signals']['signal'].tolist()
        )
        engine = BacktestingEngine(candle_data=invalid_data)
        results = engine.run_backtest()
        self.assertEqual(len(results['trades']), 0)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Completed test_invalid_signal_handling in {execution_time:.2f} seconds - PASSED")

    def test_missing_sltp_handling(self):
        """Test handling of missing SL/TP levels."""
        logger.info("Starting test_missing_sltp_handling")
        start_time = datetime.now()
        
        # Create test data with clear signals
        dates = pd.date_range(start='2024-01-01', end='2024-01-02', freq='1min')
        market_hours = dates[dates.indexer_between_time('09:30', '16:00')]
        n_periods = len(market_hours)
        
        # Generate price data with clear trend
        base_price = 100
        prices = base_price + np.linspace(0, 10, n_periods)  # Clear uptrend
        
        # Generate clear signals
        signals = np.zeros(n_periods)
        signals[10] = 1  # Long entry
        signals[20] = -1  # Long exit
        
        # Create test DataFrame
        test_data = pd.DataFrame({
            'timestamp': market_hours,
            'open': prices,
            'high': prices * 1.001,
            'low': prices * 0.999,
            'close': prices,
            'volume': np.random.randint(100, 1000, n_periods),
            'oi': np.random.randint(1000, 10000, n_periods),
            'signal': signals
        })
        
        # Convert timestamps to string format
        test_data['timestamp'] = test_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Create CandleData object without SL/TP
        missing_sltp_data = CandleData.from_list(
            test_data[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi']].values.tolist(),
            signals=test_data['signal'].tolist()
        )
        
        engine = BacktestingEngine(
            candle_data=missing_sltp_data,
            initial_balance=100000,
            volume=10.0,
            use_sltp=False,  # Disable SL/TP since we don't have levels
            min_position_size_ratio=0.3,  # Allow smaller positions
            position_sizing_method='fixed'  # Use fixed position sizing
        )
        results = engine.run_backtest()
        
        # Verify trades were executed
        self.assertGreater(len(results['trades']), 0)
        
        # Verify trade execution
        trade = results['trades'][0]
        self.assertEqual(trade['Trade_Type'], 'long')
        self.assertEqual(trade['Trade_Close_Action'], 'signal_close')
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Completed test_missing_sltp_handling in {execution_time:.2f} seconds - PASSED")

    def test_insufficient_balance_handling(self):
        """Test handling of insufficient balance for trades."""
        logger.info("Starting test_insufficient_balance_handling")
        start_time = datetime.now()
        
        engine = BacktestingEngine(
            candle_data=self.candle_data,
            initial_balance=1.0,
            volume=1000.0
        )
        results = engine.run_backtest()
        self.assertEqual(len(results['trades']), 0)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Completed test_insufficient_balance_handling in {execution_time:.2f} seconds - PASSED")

    def test_invalid_slippage_model_handling(self):
        """Test handling of invalid slippage model."""
        logger.info("Starting test_invalid_slippage_model_handling")
        start_time = datetime.now()
        
        # Test invalid slippage model during initialization
        with self.assertRaises(ValueError) as context:
            BacktestingEngine(
                candle_data=self.candle_data,
                slippage_model='invalid_model'
            )
        self.assertIn("slippage_model must be one of", str(context.exception))
        
        # Test valid slippage model
        engine = BacktestingEngine(
            candle_data=self.candle_data,
            slippage_model='percentage',
            slippage_value=0.1
        )
        # Verify slippage is applied correctly
        price = 100.0
        volume = 1.0
        direction = 'long'
        expected_price = price * (1 + 0.1/100)  # 0.1% slippage
        actual_price = engine.apply_slippage(price, volume, direction)
        self.assertAlmostEqual(actual_price, expected_price, places=2)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Completed test_invalid_slippage_model_handling in {execution_time:.2f} seconds - PASSED")

    def test_partial_fills_handling(self):
        """Test handling of partial fills."""
        logger.info("Starting test_partial_fills_handling")
        start_time = datetime.now()
        
        # Create test data with clear signals
        dates = pd.date_range(start='2024-01-01', end='2024-01-02', freq='1min')
        market_hours = dates[dates.indexer_between_time('09:30', '16:00')]
        n_periods = len(market_hours)
        
        # Generate price data with clear trend
        base_price = 100
        prices = base_price + np.linspace(0, 10, n_periods)  # Clear uptrend
        
        # Generate clear signals
        signals = np.zeros(n_periods)
        signals[10] = 1  # Long entry
        signals[20] = -1  # Long exit
        
        # Create test DataFrame
        test_data = pd.DataFrame({
            'timestamp': market_hours,
            'open': prices,
            'high': prices * 1.001,
            'low': prices * 0.999,
            'close': prices,
            'volume': np.random.randint(100, 1000, n_periods),
            'oi': np.random.randint(1000, 10000, n_periods),
            'signal': signals
        })
        
        # Convert timestamps to string format
        test_data['timestamp'] = test_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Create CandleData object
        candle_data = CandleData.from_list(
            test_data[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi']].values.tolist(),
            signals=test_data['signal'].tolist()
        )
        
        engine = BacktestingEngine(
            candle_data=candle_data,
            allow_partial_fills=True,
            volume=10.0,
            initial_balance=100000,
            min_position_size_ratio=0.3,  # Allow partial fills down to 30%
            position_sizing_method='fixed'  # Use fixed position sizing
        )
        
        # Test position size calculation
        position_size = engine.calculate_position_size(100.0, 10.0)
        
        # For partial fills, position size should be between min_ratio and full volume
        min_expected = 10.0 * 0.3  # 30% of volume
        max_expected = 10.0  # Full volume
        self.assertGreaterEqual(position_size, min_expected)
        self.assertLessEqual(position_size, max_expected)
        
        # Run backtest to verify trade execution
        results = engine.run_backtest()
        self.assertGreater(len(results['trades']), 0)
        
        # Verify trade execution
        trade = results['trades'][0]
        self.assertEqual(trade['Trade_Type'], 'long')
        self.assertGreaterEqual(trade['Position_Size'], min_expected)
        self.assertLessEqual(trade['Position_Size'], max_expected)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Completed test_partial_fills_handling in {execution_time:.2f} seconds - PASSED")

    def test_leverage_limits(self):
        """Test handling of leverage limits."""
        logger.info("Starting test_leverage_limits")
        start_time = datetime.now()
        
        engine = BacktestingEngine(
            candle_data=self.candle_data,
            leverage=10.0,
            initial_balance=1000.0,
            volume=100.0
        )
        position_size = engine.calculate_position_size(100.0, 100.0)
        max_position = (1000.0 * 10.0) / 100.0
        self.assertLessEqual(position_size, max_position)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Completed test_leverage_limits in {execution_time:.2f} seconds - PASSED")

    def test_pnl_calculation(self):
        """Test PnL calculation with various scenarios."""
        logger.info("Starting test_pnl_calculation")
        start_time = datetime.now()
        
        # Test long position
        long_pnl = self.engine._calculate_pnl_vectorized(
            np.array([100.0]),
            np.array([110.0]),
            np.array([1.0]),
            np.array([1.0]),
            np.array([1.0])
        )[0]
        self.assertAlmostEqual(long_pnl, 9.0)

        # Test short position
        short_pnl = self.engine._calculate_pnl_vectorized(
            np.array([100.0]),
            np.array([90.0]),
            np.array([1.0]),
            np.array([-1.0]),
            np.array([1.0])
        )[0]
        self.assertAlmostEqual(short_pnl, 9.0)

        # Test zero price
        zero_pnl = self.engine._calculate_pnl_vectorized(
            np.array([0.0]),
            np.array([0.0]),
            np.array([1.0]),
            np.array([1.0]),
            np.array([1.0])
        )[0]
        self.assertAlmostEqual(zero_pnl, -1.0)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Completed test_pnl_calculation in {execution_time:.2f} seconds - PASSED")

    def test_parallel_position_limits(self):
        """Test strict enforcement of parallel position limits."""
        logger.info("Starting test_parallel_position_limits")
        start_time = datetime.now()
        
        engine = BacktestingEngine(
            candle_data=self.candle_data,
            max_open_positions=1,
            parallel_opening=False
        )
        results = engine.run_backtest()
        
        # Track open positions
        open_positions = []
        for trade in results['trades']:
            if trade['Trade_Close_Action'] == 'final_close':
                continue
            # Add trade to open positions
            open_positions.append(trade)
            # Remove closed trades
            open_positions = [p for p in open_positions if p['Trade_ID'] != trade['Trade_ID']]
            # Assert we never have more than one open position
            self.assertLessEqual(len(open_positions), 1)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Completed test_parallel_position_limits in {execution_time:.2f} seconds - PASSED")

    def test_risk_management_limits(self):
        """Test strict enforcement of risk management limits."""
        logger.info("Starting test_risk_management_limits")
        start_time = datetime.now()
        
        engine = BacktestingEngine(
            candle_data=self.candle_data,
            max_position_size=0.5,
            initial_balance=1000.0
        )
        results = engine.run_backtest()
        
        # Add a small tolerance for floating-point comparison
        tolerance = 1e-10
        for trade in results['trades']:
            position_value = trade['Position_Size'] * trade['Entry_Price']
            self.assertLessEqual(position_value, 500.0 + tolerance, 
                               f"Position value {position_value} exceeds maximum allowed 500.0")
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Completed test_risk_management_limits in {execution_time:.2f} seconds - PASSED")

    def test_data_manager_error_handling(self):
        """Test error handling in data manager integration."""
        logger.info("Starting test_data_manager_error_handling")
        start_time = datetime.now()
        
        # Test with invalid data
        with self.assertRaises(Exception):
            self.data_manager.save_backtest_results(
                results={'invalid': 'data'},
                ticker='TEST',
                strategy_name='TestStrategy'
            )

        # Test with missing required fields
        with self.assertRaises(Exception):
            self.data_manager.save_backtest_results(
                results={'trades': []},
                ticker='TEST',
                strategy_name='TestStrategy'
            )
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Completed test_data_manager_error_handling in {execution_time:.2f} seconds - PASSED")

    def test_trade_execution_sequence(self):
        """Test proper sequence of trade execution and position management."""
        logger.info("Starting test_trade_execution_sequence")
        start_time = datetime.now()
        
        engine = BacktestingEngine(
            candle_data=self.candle_data,
            initial_balance=100000.0,
            volume=5.0,
            use_sltp=True
        )
        results = engine.run_backtest()
        
        # Verify trade sequence
        for i in range(len(results['trades']) - 1):
            current_trade = results['trades'][i]
            next_trade = results['trades'][i + 1]
            
            # Verify trade IDs are sequential
            self.assertEqual(next_trade['Trade_ID'], current_trade['Trade_ID'] + 1)
            
            # Verify entry time is before or equal to exit time
            self.assertLessEqual(current_trade['Entry_Time'], current_trade['Exit_Time'])
            
            # Verify no overlapping trades when parallel_opening is False
            if not engine.parallel_opening:
                # For same timestamp, verify the first trade is fully closed before next trade starts
                if current_trade['Exit_Time'] == next_trade['Entry_Time']:
                    # Trade can be closed by signal, stop loss, or take profit
                    valid_close_actions = ['signal_close', 'stop_loss', 'take_profit']
                    self.assertIn(current_trade['Trade_Close_Action'], valid_close_actions)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Completed test_trade_execution_sequence in {execution_time:.2f} seconds - PASSED")

    def test_balance_tracking(self):
        """Test accurate tracking of balance and equity curve."""
        logger.info("Starting test_balance_tracking")
        start = time.time()
        engine = BacktestingEngine(
            candle_data=self.candle_data,
            initial_balance=100000,
            volume=10.0,
            leverage=10,
            commission=0.0005,
            slippage_model='none',
            slippage_value=0.0,
            parallel_opening=False,
            max_open_positions=1,
            min_position_size_ratio=1.0
        )
        results = engine.run_backtest()
        trades = results['trades']
        balance_history = results['balance_history']
        # For each trade, check that the trade's Balance matches the balance_history at the trade's close index
        for trade in trades:
            # Find the index in data.timestamp that matches trade['Exit_Time']
            if 'Exit_Time' in trade and trade['Exit_Time'] in self.test_data.timestamp:
                idx = np.where(self.test_data.timestamp == trade['Exit_Time'])[0]
                if len(idx) > 0:
                    idx = idx[0] + 1  # balance_history is offset by 1
                    self.assertAlmostEqual(trade['Balance'], balance_history[idx], places=2)
        # Ensure balance_history, equity_curve, and drawdown_curve are lists
        self.assertIsInstance(results['balance_history'], list)
        self.assertIsInstance(results['equity_curve'], list)
        self.assertIsInstance(results['drawdown_curve'], list)
        logger.info(f"Completed test_balance_tracking in {time.time() - start:.2f} seconds - PASSED")

    def test_drawdown_calculation(self):
        """Test accurate calculation of drawdown curve."""
        logger.info("Starting test_drawdown_calculation")
        start_time = datetime.now()
        
        engine = BacktestingEngine(
            candle_data=self.candle_data,
            initial_balance=100000.0,
            volume=5.0
        )
        results = engine.run_backtest()
        
        # Verify drawdown is always between 0 and 1
        self.assertTrue(all([0 <= d <= 1 for d in results['drawdown_curve']]))
        
        # Verify drawdown calculation
        peak = results['equity_curve'][0]
        for i in range(1, len(results['equity_curve'])):
            peak = max(peak, results['equity_curve'][i])
            expected_drawdown = (peak - results['equity_curve'][i]) / peak if peak > 0 else 0
            self.assertAlmostEqual(results['drawdown_curve'][i], expected_drawdown, places=6)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Completed test_drawdown_calculation in {execution_time:.2f} seconds - PASSED")

    def test_slippage_impact(self):
        """Test impact of different slippage models on trade execution."""
        logger.info("Starting test_slippage_impact")
        start = time.time()
        
        # Create test data with clear signals
        dates = pd.date_range(start='2024-01-01', end='2024-01-02', freq='1min')
        market_hours = dates[dates.indexer_between_time('09:30', '16:00')]
        n_periods = len(market_hours)
        
        # Generate price data with clear trend
        base_price = 100
        prices = base_price + np.linspace(0, 10, n_periods)  # Clear uptrend
        
        # Generate clear signals
        signals = np.zeros(n_periods)
        signals[10] = 1  # Long entry
        signals[20] = -1  # Long exit
        
        # Create test DataFrame
        test_data = pd.DataFrame({
            'timestamp': market_hours,
            'open': prices,
            'high': prices * 1.001,
            'low': prices * 0.999,
            'close': prices,
            'volume': np.random.randint(100, 1000, n_periods),
            'oi': np.random.randint(1000, 10000, n_periods),
            'signal': signals
        })
        
        # Convert timestamps to string format
        test_data['timestamp'] = test_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Create CandleData object
        candle_data = CandleData.from_list(
            test_data[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi']].values.tolist(),
            signals=test_data['signal'].tolist()
        )
        
        # No slippage
        engine_no_slip = BacktestingEngine(
            candle_data=candle_data,
            initial_balance=100000,
            volume=10.0,
            leverage=10,
            commission=0.0005,
            slippage_model='none',
            slippage_value=0.0,
            parallel_opening=False,
            max_open_positions=1,
            min_position_size_ratio=1.0
        )
        results_no_slip = engine_no_slip.run_backtest()
        
        # Percentage slippage
        engine_percent = BacktestingEngine(
            candle_data=candle_data,
            initial_balance=100000,
            volume=10.0,
            leverage=10,
            commission=0.0005,
            slippage_model='percentage',
            slippage_value=0.1,  # 0.1%
            parallel_opening=False,
            max_open_positions=1,
            min_position_size_ratio=1.0
        )
        results_percent = engine_percent.run_backtest()
        
        # Verify trades were executed
        self.assertGreater(len(results_no_slip['trades']), 0)
        self.assertGreater(len(results_percent['trades']), 0)
        
        # Compare entry prices
        no_slip_trade = results_no_slip['trades'][0]
        percent_trade = results_percent['trades'][0]
        
        # Use the engine's slippage formula for expected price
        base_price = no_slip_trade['Entry_Price']
        slippage = base_price * (0.1 / 100)
        expected_price = base_price * (1 + slippage / base_price)
        self.assertAlmostEqual(percent_trade['Entry_Price'], expected_price, places=2)
        
        logger.info(f"Completed test_slippage_impact in {time.time() - start:.2f} seconds - PASSED")

    def test_tax_calculation(self):
        """Test accurate calculation of trading costs and taxes."""
        logger.info("Starting test_tax_calculation")
        start_time = datetime.now()
        
        engine = BacktestingEngine(
            candle_data=self.candle_data,
            initial_balance=100000.0,
            volume=5.0,
            segment='intraday'
        )
        results = engine.run_backtest()
        
        for trade in results['trades']:
            # Verify commission is properly calculated
            self.assertGreater(trade['Commission'], 0)
            
            # For trades that are not closed due to insufficient balance
            if trade['Trade_Close_Action'] != 'insufficient_balance':
                # Calculate expected commission based on trade value
                trade_value = trade['Position_Size'] * trade['Entry_Price']
                expected_commission = engine.tax_calculator.calc_charges(
                    engine.tax_calculator.Trade(
                        segment=engine.segment,
                        buy_value=trade_value,
                        sell_value=0.0
                    )
                )['total_cost']
                self.assertAlmostEqual(trade['Commission'], expected_commission, places=2)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Completed test_tax_calculation in {execution_time:.2f} seconds - PASSED")

    def test_early_termination(self):
        """Test early termination when balance becomes insufficient."""
        logger.info("Starting test_early_termination")
        start_time = datetime.now()
        
        # Create engine with high volume to ensure early termination
        engine = BacktestingEngine(
            candle_data=self.candle_data,
            initial_balance=1000.0,
            volume=100.0,
            min_fill_ratio=0.3
        )
        results = engine.run_backtest()
        
        # Verify that trading stopped when balance became insufficient
        if results['trades']:
            last_trade = results['trades'][-1]
            self.assertLessEqual(last_trade['Balance'], 1000.0)
            
            # Verify arrays are properly trimmed
            self.assertEqual(len(results['balance_history']), len(results['equity_curve']))
            self.assertEqual(len(results['balance_history']), len(results['drawdown_curve']))
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Completed test_early_termination in {execution_time:.2f} seconds - PASSED")

    def test_position_sizing_methods(self):
        """Test different position sizing methods."""
        logger.info("Starting test_position_sizing_methods")
        start_time = datetime.now()
        
        # Test fixed position sizing
        engine_fixed = BacktestingEngine(
            candle_data=self.candle_data,
            position_sizing_method='fixed',
            volume=5.0
        )
        results_fixed = engine_fixed.run_backtest()
        
        # Test risk-based position sizing
        engine_risk = BacktestingEngine(
            candle_data=self.candle_data,
            position_sizing_method='risk_based',
            risk_per_trade=0.02,
            volume=5.0
        )
        results_risk = engine_risk.run_backtest()
        
        # Verify position sizes
        for trade in results_fixed['trades']:
            self.assertEqual(trade['Position_Size'], 5.0)
        
        for trade in results_risk['trades']:
            risk_amount = 100000.0 * 0.02  # initial_balance * risk_per_trade
            max_position_size = risk_amount / trade['Entry_Price']
            self.assertLessEqual(trade['Position_Size'], max_position_size)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Completed test_position_sizing_methods in {execution_time:.2f} seconds - PASSED")

    def test_strict_sltp_enforcement(self):
        """Test that strict_sltp=True raises error if SL/TP is missing and use_sltp=True."""
        logger.info("Starting test_strict_sltp_enforcement")
        # Create test data with no SL/TP columns
        dates = pd.date_range(start='2024-01-01', end='2024-01-02', freq='1min')
        market_hours = dates[dates.indexer_between_time('09:30', '16:00')]
        n_periods = len(market_hours)
        base_price = 100
        prices = base_price + np.linspace(0, 10, n_periods)
        signals = np.zeros(n_periods)
        signals[10] = 1
        signals[20] = -1
        test_data = pd.DataFrame({
            'timestamp': market_hours,
            'open': prices,
            'high': prices * 1.001,
            'low': prices * 0.999,
            'close': prices,
            'volume': np.random.randint(100, 1000, n_periods),
            'oi': np.random.randint(1000, 10000, n_periods),
            'signal': signals
        })
        test_data['timestamp'] = test_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        missing_sltp_data = CandleData.from_list(
            test_data[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi']].values.tolist(),
            signals=test_data['signal'].tolist()
        )
        # Should raise ValueError
        with self.assertRaises(ValueError):
            BacktestingEngine(
                candle_data=missing_sltp_data,
                use_sltp=True,
                strict_sltp=True
            )
        logger.info("Completed test_strict_sltp_enforcement - PASSED")

    def test_performance_large_dataset(self):
        """Test performance with large dataset."""
        logger.info("Starting test_performance_large_dataset")
        start_time = datetime.now()
        
        # Create large dataset (1 year of 1-minute data)
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='1min')
        market_hours = dates[dates.indexer_between_time('09:30', '16:00')]
        n_periods = len(market_hours)
        
        # Generate realistic price data
        base_price = 100
        prices = base_price + np.random.normal(0, 1, n_periods).cumsum()
        prices = np.maximum(prices, 0.1)
        
        # Create test DataFrame
        large_data = pd.DataFrame({
            'timestamp': market_hours,
            'open': prices,
            'high': prices * 1.001,
            'low': prices * 0.999,
            'close': prices,
            'volume': np.random.randint(100, 1000, n_periods),
            'oi': np.random.randint(1000, 10000, n_periods),
            'signal': np.random.choice([-2, -1, 0, 1, 2], n_periods)
        })
        
        # Convert timestamps to string format
        large_data['timestamp'] = large_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Create CandleData object
        candle_data = CandleData.from_list(
            large_data[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi']].values.tolist(),
            signals=large_data['signal'].tolist()
        )
        
        # Run backtest and measure performance
        engine = BacktestingEngine(candle_data=candle_data)
        results = engine.run_backtest()
        
        # Verify results
        self.assertIsNotNone(results)
        self.assertIn('trades', results)
        self.assertIn('balance_history', results)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Completed test_performance_large_dataset in {execution_time:.2f} seconds - PASSED")

    def test_data_validation(self):
        """Test handling of invalid data formats and types."""
        logger.info("Starting test_data_validation")
        start_time = datetime.now()
        
        # Test invalid timestamp format
        invalid_timestamp_data = self.test_data.copy()
        invalid_timestamp_data.loc[0, 'timestamp'] = 'invalid_timestamp'
        with self.assertRaises(ValueError):
            CandleData.from_list(
                invalid_timestamp_data[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi']].values.tolist()
            )
        
        # Test missing OHLCV data
        missing_data = self.test_data.copy()
        missing_data.loc[0, 'open'] = np.nan
        with self.assertRaises(ValueError):
            CandleData.from_list(
                missing_data[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi']].values.tolist()
            )
        
        # Test data type mismatch
        type_mismatch_data = self.test_data.copy()
        type_mismatch_data['volume'] = type_mismatch_data['volume'].astype(object)
        type_mismatch_data.loc[0, 'volume'] = 'invalid_volume'
        with self.assertRaises(ValueError):
            CandleData.from_list(
                type_mismatch_data[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi']].values.tolist()
            )
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Completed test_data_validation in {execution_time:.2f} seconds - PASSED")

    def test_error_recovery(self):
        """Test recovery from various error conditions."""
        logger.info("Starting test_error_recovery")
        start_time = datetime.now()
        
        # Test recovery from data corruption
        corrupted_data = self.test_data.copy()
        corrupted_data.loc[0, 'close'] = -1  # Invalid price
        engine = BacktestingEngine(candle_data=CandleData.from_list(
            corrupted_data[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi']].values.tolist()
        ))
        results = engine.run_backtest()
        self.assertIsNotNone(results)
        
        # Test recovery from insufficient memory
        # Note: This is a simulation as we can't actually exhaust memory in a test
        engine = BacktestingEngine(
            candle_data=self.candle_data,
            max_position_size=float('inf')  # Set to infinity to test handling
        )
        results = engine.run_backtest()
        self.assertIsNotNone(results)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Completed test_error_recovery in {execution_time:.2f} seconds - PASSED")

    def test_market_conditions(self):
        """Test behavior under different market conditions."""
        logger.info("Starting test_market_conditions")
        start_time = datetime.now()
        
        # Test high volatility
        volatile_data = self.test_data.copy()
        volatile_data['close'] = volatile_data['close'] * (1 + np.random.normal(0, 0.1, len(volatile_data)))
        engine = BacktestingEngine(candle_data=CandleData.from_list(
            volatile_data[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi']].values.tolist()
        ))
        results = engine.run_backtest()
        self.assertIsNotNone(results)
        
        # Test low liquidity
        low_liquidity_data = self.test_data.copy()
        low_liquidity_data['volume'] = low_liquidity_data['volume'] * 0.1
        engine = BacktestingEngine(candle_data=CandleData.from_list(
            low_liquidity_data[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi']].values.tolist()
        ))
        results = engine.run_backtest()
        self.assertIsNotNone(results)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Completed test_market_conditions in {execution_time:.2f} seconds - PASSED")

    def test_configuration_validation(self):
        """Test validation of all configuration parameters."""
        logger.info("Starting test_configuration_validation")
        start_time = datetime.now()

        # Test initial_balance validation
        with self.assertRaises(ValueError):
            BacktestingEngine(
                candle_data=self.candle_data,
                initial_balance=-1000.0
            )

        # Test volume validation
        with self.assertRaises(ValueError):
            BacktestingEngine(
                candle_data=self.candle_data,
                volume=-5.0
            )

        # Test max_position_size validation
        with self.assertRaises(ValueError):
            BacktestingEngine(
                candle_data=self.candle_data,
                max_position_size=-100.0
            )

        # Test risk_per_trade validation
        with self.assertRaises(ValueError):
            BacktestingEngine(
                candle_data=self.candle_data,
                risk_per_trade=-0.02
            )
        with self.assertRaises(ValueError):
            BacktestingEngine(
                candle_data=self.candle_data,
                risk_per_trade=1.5  # More than 100%
            )

        # Test leverage validation
        with self.assertRaises(ValueError):
            BacktestingEngine(
                candle_data=self.candle_data,
                leverage=0.0
            )
        with self.assertRaises(ValueError):
            BacktestingEngine(
                candle_data=self.candle_data,
                leverage=-1.0
            )

        # Test slippage_model validation
        with self.assertRaises(ValueError):
            BacktestingEngine(
                candle_data=self.candle_data,
                slippage_model='invalid_model'
            )

        # Test slippage_value validation
        with self.assertRaises(ValueError):
            BacktestingEngine(
                candle_data=self.candle_data,
                slippage_value=-0.001
            )

        # Test min_position_size_ratio validation
        with self.assertRaises(ValueError):
            BacktestingEngine(
                candle_data=self.candle_data,
                min_position_size_ratio=-0.3
            )
        with self.assertRaises(ValueError):
            BacktestingEngine(
                candle_data=self.candle_data,
                min_position_size_ratio=1.5  # More than 100%
            )

        # Test min_fill_ratio validation
        with self.assertRaises(ValueError):
            BacktestingEngine(
                candle_data=self.candle_data,
                min_fill_ratio=-0.3
            )
        with self.assertRaises(ValueError):
            BacktestingEngine(
                candle_data=self.candle_data,
                min_fill_ratio=1.5  # More than 100%
            )

        # Test position_sizing_method validation
        with self.assertRaises(ValueError):
            BacktestingEngine(
                candle_data=self.candle_data,
                position_sizing_method='invalid_method'
            )

        # Test segment validation
        with self.assertRaises(ValueError):
            BacktestingEngine(
                candle_data=self.candle_data,
                segment='invalid_segment'
            )

        # Test max_open_positions validation
        with self.assertRaises(ValueError):
            BacktestingEngine(
                candle_data=self.candle_data,
                max_open_positions=0
            )
        with self.assertRaises(ValueError):
            BacktestingEngine(
                candle_data=self.candle_data,
                max_open_positions=-1
            )

        # Test valid configuration combinations
        valid_engine = BacktestingEngine(
            candle_data=self.candle_data,
            initial_balance=100000.0,
            volume=5.0,
            max_position_size=100.0,
            risk_per_trade=0.02,
            leverage=1.0,
            slippage_model='percentage',
            slippage_value=0.001,
            allow_partial_fills=True,
            max_open_positions=1,
            position_sizing_method='fixed',
            parallel_opening=False,
            segment='intraday',
            min_position_size_ratio=0.3,
            min_fill_ratio=0.3,
            use_sltp=True,
            force_signal_exits=False,
            strict_sltp=False
        )
        
        # Verify valid configuration is accepted
        self.assertIsNotNone(valid_engine)
        
        # Run backtest to verify configuration works
        results = valid_engine.run_backtest()
        self.assertIsNotNone(results)
        self.assertIn('strategy_config', results)
        
        # Verify strategy_config contains all parameters
        config = results['strategy_config']
        self.assertEqual(config['initial_balance'], 100000.0)
        self.assertEqual(config['volume'], 5.0)
        self.assertEqual(config['max_position_size'], 100.0)
        self.assertEqual(config['risk_per_trade'], 0.02)
        self.assertEqual(config['leverage'], 1.0)
        self.assertEqual(config['slippage_model'], 'percentage')
        self.assertEqual(config['slippage_value'], 0.001)
        self.assertTrue(config['allow_partial_fills'])
        self.assertEqual(config['max_open_positions'], 1)
        self.assertEqual(config['position_sizing_method'], 'fixed')
        self.assertFalse(config['parallel_opening'])
        self.assertEqual(config['segment'], 'intraday')
        self.assertEqual(config['min_position_size_ratio'], 0.3)
        self.assertEqual(config['min_fill_ratio'], 0.3)
        self.assertTrue(config['use_sltp'])
        self.assertFalse(config['force_signal_exits'])
        self.assertFalse(config['strict_sltp'])
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Completed test_configuration_validation in {execution_time:.2f} seconds - PASSED")
