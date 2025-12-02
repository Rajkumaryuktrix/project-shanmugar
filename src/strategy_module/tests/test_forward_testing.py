import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
import sys
import os
import logging
import time
import tempfile
import shutil
import json

# Add the src directory to the Python path
src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from strategy_module.utils.forward_testing import (
    ForwardConfig,
    ForwardTestingEngine,
    SimulatorConfig,
    create_forward_testing_engine
)
from strategy_module.utils.backward_testing import (
    BacktestingEngine,
    CandleData,
    create_backtesting_engine
)

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler with formatting
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class TestForwardTesting(unittest.TestCase):
    """Test cases for forward testing module."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data that will be used across all test cases."""
        logger.info("Setting up test data for all test methods")
        cls.start_time = datetime.now()
        
        # Create sample candle data
        timestamps = np.array([
            int((datetime.now() - timedelta(days=i)).timestamp())
            for i in range(100)
        ])[::-1]  # Reverse to get ascending order
        
        cls.sample_candle_data = CandleData(
            timestamp=timestamps,
            open=np.random.uniform(100, 200, 100),
            high=np.random.uniform(200, 300, 100),
            low=np.random.uniform(50, 100, 100),
            close=np.random.uniform(100, 200, 100),
            volume=np.random.uniform(1000, 5000, 100),
            oi=np.random.uniform(10000, 50000, 100),
            signal=np.zeros(100)
        )
        
        # Create backtesting engine
        cls.backtest_engine = create_backtesting_engine(
            cls.sample_candle_data,
            initial_balance=100000,
            volume=10,
            risk_per_trade=0.02
        )
        
        # Create base simulator config
        cls.base_simulator_config = {
            'type': 'monte_carlo',
            'enabled': True,
            'parameters': {
                'num_paths': 100,
                'forecast_periods': 30,
                'volatility_window': 20,
                'drift_window': 20,
                'use_historical_vol': True,
                'use_historical_drift': True,
                'min_price': 0.0,
                'max_price': float('inf'),
                'volume_scaling': 1.0,
                'min_volume': 0.0,
                'max_volume': float('inf')
            }
        }
        
        # Create base forward testing config
        cls.config = ForwardConfig(
            candle_data=cls.sample_candle_data,
            backtest_engine=cls.backtest_engine,
            simulators=[cls.base_simulator_config],
            strategy_name='moving_average',
            strategy_params={
                'fast_period': 5,
                'slow_period': 10
            }
        )
        
        # Create edge case data
        cls.edge_case_data = {
            'empty_data': CandleData(
                timestamp=np.array([]),
                open=np.array([]),
                high=np.array([]),
                low=np.array([]),
                close=np.array([]),
                volume=np.array([]),
                oi=np.array([]),
                signal=np.array([])
            ),
            'single_candle': CandleData(
                timestamp=np.array([int(datetime.now().timestamp())]),
                open=np.array([100.0]),
                high=np.array([110.0]),
                low=np.array([90.0]),
                close=np.array([105.0]),
                volume=np.array([1000.0]),
                oi=np.array([10000.0]),
                signal=np.array([0.0])
            ),
            'extreme_values': CandleData(
                timestamp=timestamps,
                open=np.full(100, 1e9),
                high=np.full(100, 1e9),
                low=np.full(100, 1e9),
                close=np.full(100, 1e9),
                volume=np.full(100, 1e9),
                oi=np.full(100, 1e9),
                signal=np.zeros(100)
            )
        }
        
        end_time = datetime.now()
        execution_time = (end_time - cls.start_time).total_seconds()
        logger.info(f"Test data setup completed in {execution_time:.2f} seconds")

    def setUp(self):
        """Set up test fixtures."""
        self.base_simulator_config = {
            'type': 'monte_carlo',
            'enabled': True,
            'parameters': {
                'num_paths': 10,
                'forecast_periods': 5,
                'volatility_window': 1,
                'drift_window': 1,
                'use_historical_vol': False,
                'use_historical_drift': False,
                'min_price': 0,
                'max_price': float('inf'),
                'volume_scaling': 1.0,
                'min_volume': 0.0,
                'max_volume': float('inf')
            }
        }
        self.config = ForwardConfig(
            candle_data=self.sample_candle_data,
            backtest_engine=self.backtest_engine,
            simulators=[self.base_simulator_config],
            strategy_name='moving_average',
            strategy_params={}
        )
        
        # Create temporary directory for test data
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up after each test method."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)

    def test_forward_config_validation(self):
        """Test ForwardConfig validation."""
        logger.info("Starting test_forward_config_validation")
        start_time = datetime.now()
        
        # Test valid configuration
        config = ForwardConfig(
            candle_data=self.sample_candle_data,
            backtest_engine=self.backtest_engine,
            simulators=[self.base_simulator_config],
            strategy_name='moving_average',
            confidence_level=0.95,
            significance_level=0.05,
            max_drawdown=0.2,
            risk_per_trade=0.02,
            max_position_size=100000
        )
        config.validate()  # Should not raise any exception
        
        # Test invalid configurations
        with self.assertRaises(ValueError):
            # Test with None candle data
            config = ForwardConfig(
                candle_data=None,
                backtest_engine=self.backtest_engine,
                simulators=[self.base_simulator_config],
                strategy_name='moving_average'
            )
            config.validate()
            
        with self.assertRaises(ValueError):
            # Test with empty simulators list
            config = ForwardConfig(
                candle_data=self.sample_candle_data,
                backtest_engine=self.backtest_engine,
                simulators=[],
                strategy_name='moving_average'
            )
            config.validate()
            
        with self.assertRaises(ValueError):
            # Test with all disabled simulators
            config = ForwardConfig(
                candle_data=self.sample_candle_data,
                backtest_engine=self.backtest_engine,
                simulators=[{
                    'type': 'monte_carlo',
                    'enabled': False,
                    'parameters': self.base_simulator_config['parameters']
                }],
                strategy_name='moving_average'
            )
            config.validate()
            
        with self.assertRaises(ValueError):
            # Test with invalid confidence level
            config = ForwardConfig(
                candle_data=self.sample_candle_data,
                backtest_engine=self.backtest_engine,
                simulators=[self.base_simulator_config],
                strategy_name='moving_average',
                confidence_level=1.5
            )
            config.validate()
            
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Completed test_forward_config_validation in {execution_time:.2f} seconds - PASSED")

    def test_simulator_config_validation(self):
        """Test SimulatorConfig validation."""
        logger.info("Starting test_simulator_config_validation")
        start_time = datetime.now()
        
        # Test valid configuration
        config = SimulatorConfig(
            name='montecarlo',
            enabled=True,
            params={
                'num_paths': 100,
                'forecast_periods': 30
            }
        )
        self.assertEqual(config.name, 'montecarlo')
        self.assertTrue(config.enabled)
        self.assertIsNotNone(config.params)
        
        # Test with minimal configuration
        config = SimulatorConfig(name='montecarlo')
        self.assertEqual(config.name, 'montecarlo')
        self.assertTrue(config.enabled)
        self.assertIsNone(config.params)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Completed test_simulator_config_validation in {execution_time:.2f} seconds - PASSED")

    def test_forward_testing_engine_initialization(self):
        """Test ForwardTestingEngine initialization."""
        logger.info("Starting test_forward_testing_engine_initialization")
        start_time = datetime.now()
        
        # Test valid initialization
        engine = create_forward_testing_engine(self.config)
        self.assertIsInstance(engine, ForwardTestingEngine)
        self.assertEqual(engine.config, self.config)
        self.assertEqual(len(engine.simulators), 0)  # Simulators not created until run
        
        # Test with invalid configuration
        with self.assertRaises(ValueError):
            invalid_config = ForwardConfig(
                candle_data=None,
                backtest_engine=self.backtest_engine,
                simulators=[self.base_simulator_config]
            )
            create_forward_testing_engine(invalid_config)
            
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Completed test_forward_testing_engine_initialization in {execution_time:.2f} seconds - PASSED")

    def test_simulator_creation(self):
        """Test simulator creation and selection."""
        logger.info("Starting test_simulator_creation")
        start_time = datetime.now()
        
        engine = create_forward_testing_engine(self.config)
        
        # Test simulator creation
        simulators = engine._create_simulators()
        self.assertEqual(len(simulators), 1)
        self.assertEqual(simulators[0].get_name(), 'MonteCarloSimulator')
        
        # Test with disabled simulator
        config = ForwardConfig(
            candle_data=self.sample_candle_data,
            backtest_engine=self.backtest_engine,
            simulators=[{
                'type': 'monte_carlo',
                'enabled': False,
                'parameters': self.base_simulator_config['parameters']
            }],
            strategy_name='moving_average'
        )
        with self.assertRaises(ValueError) as context:
            create_forward_testing_engine(config)
        self.assertIn("No valid simulators could be created", str(context.exception))
        
        # Test with unknown simulator type
        config = ForwardConfig(
            candle_data=self.sample_candle_data,
            backtest_engine=self.backtest_engine,
            simulators=[{
                'type': 'unknown',
                'enabled': True,
                'parameters': {}
            }],
            strategy_name='moving_average'
        )
        with self.assertRaises(ValueError) as context:
            engine = create_forward_testing_engine(config)
            engine._create_simulators()
        self.assertIn("No valid simulators could be created", str(context.exception))
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Completed test_simulator_creation in {execution_time:.2f} seconds - PASSED")

    def test_path_generation(self):
        """Test path generation functionality."""
        logger.info("Starting test_path_generation")
        start_time = datetime.now()
        
        engine = create_forward_testing_engine(self.config)
        engine.simulators = engine._create_simulators()
        
        # Test path generation
        paths = engine._generate_paths()
        self.assertIsInstance(paths, dict)
        self.assertGreater(len(paths), 0)
        
        # Verify path data structure
        for path_name, path_data in paths.items():
            self.assertIsInstance(path_data, CandleData)
            self.assertEqual(len(path_data.timestamp), self.base_simulator_config['parameters']['forecast_periods'])
            self.assertEqual(len(path_data.close), self.base_simulator_config['parameters']['forecast_periods'])
            self.assertEqual(len(path_data.volume), self.base_simulator_config['parameters']['forecast_periods'])
            
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Completed test_path_generation in {execution_time:.2f} seconds - PASSED")

    def test_backtest_execution(self):
        """Test backtest execution on generated paths."""
        logger.info("Starting test_backtest_execution")
        start_time = datetime.now()
        
        engine = create_forward_testing_engine(self.config)
        engine.simulators = engine._create_simulators()
        paths = engine._generate_paths()
        
        # Test backtest execution
        results = engine._run_backtests(paths)
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), len(paths))
        
        # Verify backtest results structure
        for scenario_name, scenario_results in results.items():
            self.assertIn('trades', scenario_results)
            self.assertIn('balance_history', scenario_results)
            self.assertIn('equity_curve', scenario_results)
            self.assertIn('drawdown_curve', scenario_results)
            
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Completed test_backtest_execution in {execution_time:.2f} seconds - PASSED")

    def test_full_forward_testing_process(self):
        """Test the complete forward testing process."""
        logger.info("Starting test_full_forward_testing_process")
        start_time = datetime.now()
        
        engine = create_forward_testing_engine(self.config)
        
        # Run forward testing
        results = engine.run()
        
        # Verify results structure
        self.assertIsInstance(results, dict)
        self.assertIn('scenarios', results)
        self.assertIn('summary', results)
        self.assertIn('config', results)
        self.assertIn('timestamp', results)
        
        # Verify summary
        summary = results['summary']
        self.assertIn('total_scenarios', summary)
        self.assertIn('scenario_names', summary)
        self.assertIn('execution_time', summary)
        self.assertGreater(summary['total_scenarios'], 0)
        self.assertGreater(len(summary['scenario_names']), 0)
        self.assertGreater(summary['execution_time'], 0)
        
        # Verify config
        config = results['config']
        self.assertIn('simulators', config)
        self.assertIn('confidence_level', config)
        self.assertIn('significance_level', config)
        self.assertIn('max_drawdown', config)
        self.assertIn('risk_per_trade', config)
        self.assertIn('max_position_size', config)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Completed test_full_forward_testing_process in {execution_time:.2f} seconds - PASSED")

    def test_error_handling(self):
        """Test error handling in various scenarios."""
        logger.info("Starting test_error_handling")
        start_time = datetime.now()
        
        engine = create_forward_testing_engine(self.config)
        
        # Test with invalid candle data
        invalid_config = ForwardConfig(
            candle_data=CandleData(
                timestamp=np.array([]),
                open=np.array([]),
                high=np.array([]),
                low=np.array([]),
                close=np.array([]),
                volume=np.array([]),
                oi=np.array([]),
                signal=np.array([])
            ),
            backtest_engine=self.backtest_engine,
            simulators=[self.base_simulator_config]
        )
        engine.config = invalid_config
        with self.assertRaises(Exception):
            engine.run()
        
        # Test with invalid simulator parameters
        invalid_sim_config = SimulatorConfig(
            name='montecarlo',
            enabled=True,
            params={
                'num_paths': -1,  # Invalid number of paths
                'forecast_periods': 0  # Invalid forecast periods
            }
        )
        invalid_config = ForwardConfig(
            candle_data=self.sample_candle_data,
            backtest_engine=self.backtest_engine,
            simulators=[invalid_sim_config]
        )
        engine.config = invalid_config
        with self.assertRaises(Exception):
            engine.run()
            
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Completed test_error_handling in {execution_time:.2f} seconds - PASSED")

    def test_end_to_end_forward_testing(self):
        """End-to-end test: simulation, backtesting, and result validation."""
        logger.info("Starting test_end_to_end_forward_testing")
        start_time = datetime.now()
        
        engine = create_forward_testing_engine(self.config)
        results = engine.run()
        
        # Validate top-level keys
        self.assertIn('scenarios', results)
        self.assertIn('summary', results)
        self.assertIn('config', results)
        self.assertIn('timestamp', results)
        
        # Validate scenarios
        self.assertIsInstance(results['scenarios'], dict)
        self.assertGreater(len(results['scenarios']), 0)
        for scenario, data in results['scenarios'].items():
            self.assertIn('trades', data)
            self.assertIn('balance_history', data)
            self.assertIn('equity_curve', data)
            self.assertIn('drawdown_curve', data)
            self.assertIsInstance(data['trades'], list)
            self.assertIsInstance(data['balance_history'], list)
            self.assertIsInstance(data['equity_curve'], list)
            self.assertIsInstance(data['drawdown_curve'], list)
            
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Completed test_end_to_end_forward_testing in {execution_time:.2f} seconds - PASSED")

    def test_empty_data_handling(self):
        """Test handling of empty candle data."""
        logger.info("Starting test_empty_data_handling")
        start_time = datetime.now()
        
        config = ForwardConfig(
            candle_data=self.edge_case_data['empty_data'],
            backtest_engine=self.backtest_engine,
            simulators=[self.base_simulator_config]
        )
        
        with self.assertRaises(Exception):
            engine = create_forward_testing_engine(config)
            engine.run()
            
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Completed test_empty_data_handling in {execution_time:.2f} seconds - PASSED")

    def test_single_candle_handling(self):
        """Test handling of single candle data."""
        logger.info("Starting test_single_candle_handling")
        start_time = datetime.now()
        
        # Create simulator config for single candle
        simulator_config = {
            'type': 'monte_carlo',
            'enabled': True,
            'parameters': {
                'num_paths': 10,
                'forecast_periods': 5,
                'volatility_window': 1,
                'drift_window': 1,
                'use_historical_vol': False,
                'use_historical_drift': False,
                'min_price': 0,
                'max_price': float('inf'),
                'volume_scaling': 1.0,
                'min_volume': 0.0,
                'max_volume': float('inf')
            }
        }
        
        # Create config with single candle data
        config = ForwardConfig(
            candle_data=self.edge_case_data['single_candle'],
            backtest_engine=self.backtest_engine,
            simulators=[simulator_config],
            strategy_name='moving_average',
            strategy_params={}
        )
        
        # Create and run forward testing engine
        engine = create_forward_testing_engine(config)
        results = engine.run()
        
        # Verify results
        self.assertIsNotNone(results)
        self.assertIn('scenarios', results)
        self.assertGreater(len(results['scenarios']), 0)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Completed test_single_candle_handling in {execution_time:.2f} seconds - PASSED")

    def test_extreme_value_handling(self):
        """Test handling of extreme values in candle data."""
        logger.info("Starting test_extreme_value_handling")
        start_time = datetime.now()
        
        config = ForwardConfig(
            candle_data=self.edge_case_data['extreme_values'],
            backtest_engine=self.backtest_engine,
            simulators=[self.base_simulator_config]
        )
        
        engine = create_forward_testing_engine(config)
        results = engine.run()
        
        self.assertIsInstance(results, dict)
        self.assertIn('scenarios', results)
        self.assertIn('summary', results)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Completed test_extreme_value_handling in {execution_time:.2f} seconds - PASSED")

    def test_multiple_simulators(self):
        """Test handling of multiple simulators."""
        logger.info("Starting test_multiple_simulators")
        start_time = datetime.now()
        
        config = ForwardConfig(
            candle_data=self.sample_candle_data,
            backtest_engine=self.backtest_engine,
            simulators=[
                {
                    'type': 'monte_carlo',
                    'enabled': True,
                    'parameters': {
                        'num_paths': 50,
                        'forecast_periods': 30,
                        'volatility_window': 20,
                        'drift_window': 20,
                        'use_historical_vol': True,
                        'use_historical_drift': True
                    }
                },
                {
                    'type': 'brownian_motion',
                    'enabled': True,
                    'parameters': {
                        'num_paths': 100,
                        'forecast_periods': 60,
                        'volatility': 0.2,
                        'drift': 0.0
                    }
                }
            ],
            strategy_name='moving_average',
            strategy_params={
                'fast_period': 5,
                'slow_period': 10
            }
        )
        
        engine = create_forward_testing_engine(config)
        results = engine.run()
        
        self.assertIsInstance(results, dict)
        self.assertIn('scenarios', results)
        self.assertGreater(len(results['scenarios']), 1)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Completed test_multiple_simulators in {execution_time:.2f} seconds - PASSED")

    def test_simulator_parameter_validation(self):
        """Test validation of simulator parameters."""
        logger.info("Starting test_simulator_parameter_validation")
        start_time = datetime.now()
        
        invalid_configs = [
            {
                'type': 'monte_carlo',
                'enabled': True,
                'parameters': {'num_paths': -1}
            },
            {
                'type': 'monte_carlo',
                'enabled': True,
                'parameters': {'forecast_periods': 0}
            },
            {
                'type': 'monte_carlo',
                'enabled': True,
                'parameters': {'volatility_window': -1}
            },
            {
                'type': 'monte_carlo',
                'enabled': True,
                'parameters': {'drift_window': -1}
            },
            {
                'type': 'monte_carlo',
                'enabled': True,
                'parameters': {'min_price': 100, 'max_price': 50}
            },
            {
                'type': 'monte_carlo',
                'enabled': True,
                'parameters': {'volume_scaling': -1}
            },
            {
                'type': 'monte_carlo',
                'enabled': True,
                'parameters': {'min_volume': 100, 'max_volume': 50}
            }
        ]
        
        for params in invalid_configs:
            config = ForwardConfig(
                candle_data=self.sample_candle_data,
                backtest_engine=self.backtest_engine,
                simulators=[params],
                strategy_name='moving_average'
            )
            
            with self.assertRaises(Exception):
                engine = create_forward_testing_engine(config)
                engine._create_simulators()
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Completed test_simulator_parameter_validation in {execution_time:.2f} seconds - PASSED")

    def test_performance_large_dataset(self):
        """Test performance with large dataset."""
        logger.info("Starting test_performance_large_dataset")
        start_time = datetime.now()
        
        # Create large dataset (1 year of daily data)
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        timestamps = np.array([int(ts.timestamp()) for ts in dates])
        
        large_data = CandleData(
            timestamp=timestamps,
            open=np.random.uniform(100, 200, len(timestamps)),
            high=np.random.uniform(200, 300, len(timestamps)),
            low=np.random.uniform(50, 100, len(timestamps)),
            close=np.random.uniform(100, 200, len(timestamps)),
            volume=np.random.uniform(1000, 5000, len(timestamps)),
            oi=np.random.uniform(10000, 50000, len(timestamps)),
            signal=np.zeros(len(timestamps))
        )
        
        config = ForwardConfig(
            candle_data=large_data,
            backtest_engine=self.backtest_engine,
            simulators=[self.base_simulator_config]
        )
        
        engine = create_forward_testing_engine(config)
        results = engine.run()
        
        self.assertIsInstance(results, dict)
        self.assertIn('scenarios', results)
        self.assertIn('summary', results)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Completed test_performance_large_dataset in {execution_time:.2f} seconds - PASSED")

    def test_result_persistence(self):
        """Test persistence of forward testing results."""
        logger.info("Starting test_result_persistence")
        start_time = datetime.now()
        
        engine = create_forward_testing_engine(self.config)
        results = engine.run()
        
        # Save results to file
        output_file = os.path.join(self.test_dir, 'forward_test_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f)
        
        # Load results from file
        with open(output_file, 'r') as f:
            loaded_results = json.load(f)
        
        # Verify loaded results
        self.assertEqual(results['summary']['total_scenarios'], 
                        loaded_results['summary']['total_scenarios'])
        self.assertEqual(len(results['scenarios']), 
                        len(loaded_results['scenarios']))
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Completed test_result_persistence in {execution_time:.2f} seconds - PASSED")

    def test_market_conditions(self):
        """Test behavior under different market conditions."""
        logger.info("Starting test_market_conditions")
        start_time = datetime.now()
        
        # Create data with different market conditions
        market_conditions = {
            'trending_up': CandleData(
                timestamp=self.sample_candle_data.timestamp,
                open=np.linspace(100, 200, 100),
                high=np.linspace(110, 210, 100),
                low=np.linspace(90, 190, 100),
                close=np.linspace(105, 205, 100),
                volume=self.sample_candle_data.volume,
                oi=self.sample_candle_data.oi,
                signal=self.sample_candle_data.signal
            ),
            'trending_down': CandleData(
                timestamp=self.sample_candle_data.timestamp,
                open=np.linspace(200, 100, 100),
                high=np.linspace(210, 110, 100),
                low=np.linspace(190, 90, 100),
                close=np.linspace(205, 105, 100),
                volume=self.sample_candle_data.volume,
                oi=self.sample_candle_data.oi,
                signal=self.sample_candle_data.signal
            ),
            'volatile': CandleData(
                timestamp=self.sample_candle_data.timestamp,
                open=self.sample_candle_data.close + np.random.normal(0, 10, 100),
                high=self.sample_candle_data.close + np.random.normal(10, 10, 100),
                low=self.sample_candle_data.close + np.random.normal(-10, 10, 100),
                close=self.sample_candle_data.close + np.random.normal(0, 10, 100),
                volume=self.sample_candle_data.volume,
                oi=self.sample_candle_data.oi,
                signal=self.sample_candle_data.signal
            )
        }
        
        for condition_name, data in market_conditions.items():
            config = ForwardConfig(
                candle_data=data,
                backtest_engine=self.backtest_engine,
                simulators=[self.base_simulator_config],
                strategy_name='moving_average'
            )
            
            engine = create_forward_testing_engine(config)
            results = engine.run()
            
            self.assertIsInstance(results, dict)
            self.assertIn('scenarios', results)
            self.assertIn('summary', results)
            
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Completed test_market_conditions in {execution_time:.2f} seconds - PASSED")

    def test_configuration_combinations(self):
        """Test different configuration combinations."""
        logger.info("Starting test_configuration_combinations")
        start_time = datetime.now()
        
        config_combinations = [
            {
                'confidence_level': 0.99,
                'significance_level': 0.01,
                'max_drawdown': 0.1,
                'risk_per_trade': 0.01,
                'max_position_size': 50000
            },
            {
                'confidence_level': 0.90,
                'significance_level': 0.10,
                'max_drawdown': 0.3,
                'risk_per_trade': 0.05,
                'max_position_size': 200000
            }
        ]
        
        for config_params in config_combinations:
            config = ForwardConfig(
                candle_data=self.sample_candle_data,
                backtest_engine=self.backtest_engine,
                simulators=[self.base_simulator_config],
                strategy_name='moving_average',
                **config_params
            )
            
            engine = create_forward_testing_engine(config)
            results = engine.run()
            
            self.assertIsInstance(results, dict)
            self.assertIn('scenarios', results)
            self.assertIn('summary', results)
            
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Completed test_configuration_combinations in {execution_time:.2f} seconds - PASSED")

    def test_strategy_integration(self):
        """Test integration of trading strategies with forward testing."""
        logger.info("Starting test_strategy_integration")
        start_time = datetime.now()
        
        # Test with Moving Average strategy
        ma_config = ForwardConfig(
            candle_data=self.sample_candle_data,
            backtest_engine=self.backtest_engine,
            simulators=[self.base_simulator_config],
            strategy_name='moving_average',
            strategy_params={
                'fast_period': 5,
                'slow_period': 10,
                'sl_atr_multiplier': 2,
                'tp_atr_multiplier': 3,
                'atr_period': 14
            }
        )
        engine = create_forward_testing_engine(ma_config)
        results = engine.run()
        self.assertIsInstance(results, dict)
        self.assertIn('scenarios', results)
        
        # Verify that trades were executed
        for scenario, data in results['scenarios'].items():
            self.assertIn('trades', data)
            self.assertGreater(len(data['trades']), 0)
        
        # Test with RSI strategy
        rsi_config = ForwardConfig(
            candle_data=self.sample_candle_data,
            backtest_engine=self.backtest_engine,
            simulators=[self.base_simulator_config],
            strategy_name='rsi',
            strategy_params={
                'rsi_period': 14,
                'overbought': 70,
                'oversold': 30,
                'sl_atr_multiplier': 2,
                'tp_atr_multiplier': 3,
                'atr_period': 14
            }
        )
        engine = create_forward_testing_engine(rsi_config)
        results = engine.run()
        self.assertIsInstance(results, dict)
        self.assertIn('scenarios', results)
        
        # Verify that trades were executed
        for scenario, data in results['scenarios'].items():
            self.assertIn('trades', data)
            self.assertGreater(len(data['trades']), 0)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Completed test_strategy_integration in {execution_time:.2f} seconds - PASSED")

    def test_signal_generation(self):
        """Test signal generation for different strategies."""
        logger.info("Starting test_signal_generation")
        start_time = datetime.now()
        
        # Test Moving Average strategy signals
        ma_config = ForwardConfig(
            candle_data=self.sample_candle_data,
            backtest_engine=self.backtest_engine,
            simulators=[self.base_simulator_config],
            strategy_name='moving_average',
            strategy_params={
                'fast_period': 5,
                'slow_period': 10,
                'sl_atr_multiplier': 2,
                'tp_atr_multiplier': 3,
                'atr_period': 14
            }
        )
        engine = create_forward_testing_engine(ma_config)
        engine.simulators = engine._create_simulators()
        paths = engine._generate_paths()
        
        # Verify signal generation
        for path_name, path_data in paths.items():
            data = {
                'time': path_data.timestamp,
                'open': path_data.open,
                'high': path_data.high,
                'low': path_data.low,
                'close': path_data.close,
                'volume': path_data.volume
            }
            signals = engine._create_strategy().generate_signals(data)
            self.assertIn('signal', signals)
            self.assertIn('sl', signals)
            self.assertIn('tp', signals)
            self.assertEqual(len(signals['signal']), len(path_data.close))
            
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Completed test_signal_generation in {execution_time:.2f} seconds - PASSED")

    def test_path_validation(self):
        """Test validation of generated price paths."""
        logger.info("Starting test_path_validation")
        start_time = datetime.now()
        
        engine = create_forward_testing_engine(self.config)
        engine.simulators = engine._create_simulators()
        paths = engine._generate_paths()
        
        for path_name, path_data in paths.items():
            # Check data consistency
            self.assertEqual(len(path_data.timestamp), len(path_data.open))
            self.assertEqual(len(path_data.open), len(path_data.high))
            self.assertEqual(len(path_data.high), len(path_data.low))
            self.assertEqual(len(path_data.low), len(path_data.close))
            self.assertEqual(len(path_data.close), len(path_data.volume))
            
            # Check price relationships
            self.assertTrue(np.all(path_data.high >= path_data.open))
            self.assertTrue(np.all(path_data.high >= path_data.close))
            self.assertTrue(np.all(path_data.low <= path_data.open))
            self.assertTrue(np.all(path_data.low <= path_data.close))
            
            # Check for invalid values
            self.assertTrue(np.all(np.isfinite(path_data.open)))
            self.assertTrue(np.all(np.isfinite(path_data.high)))
            self.assertTrue(np.all(np.isfinite(path_data.low)))
            self.assertTrue(np.all(np.isfinite(path_data.close)))
            self.assertTrue(np.all(np.isfinite(path_data.volume)))
            
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Completed test_path_validation in {execution_time:.2f} seconds - PASSED")

    def test_result_statistics(self):
        """Test calculation of result statistics."""
        logger.info("Starting test_result_statistics")
        start_time = datetime.now()
        
        engine = create_forward_testing_engine(self.config)
        results = engine.run()
        
        # Verify summary statistics
        summary = results['summary']
        self.assertIn('total_scenarios', summary)
        self.assertIn('scenario_names', summary)
        self.assertIn('total_trades', summary)
        self.assertIn('total_pnl', summary)
        self.assertIn('average_pnl_per_trade', summary)
        self.assertIn('execution_time', summary)
        
        # Verify statistics calculations
        total_trades = sum(
            len(scenario['trades'])
            for scenario in results['scenarios'].values()
        )
        self.assertEqual(summary['total_trades'], total_trades)
        
        total_pnl = sum(
            sum(trade['PnL'] for trade in scenario['trades'])
            for scenario in results['scenarios'].values()
        )
        self.assertAlmostEqual(summary['total_pnl'], total_pnl)
        
        if total_trades > 0:
            expected_avg_pnl = total_pnl / total_trades
            self.assertAlmostEqual(summary['average_pnl_per_trade'], expected_avg_pnl)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Completed test_result_statistics in {execution_time:.2f} seconds - PASSED")

    def test_config_file_loading(self):
        """Test loading of configuration from JSON files."""
        logger.info("Starting test_config_file_loading")
        start_time = datetime.now()
        
        # Create test config files
        strategy_config = {
            'moving_average': {
                'module': 'strategy_module.strategies.moving_average_strategy',
                'class': 'MovingAverageStrategy',
                'default_params': {
                    'fast_period': 5,
                    'slow_period': 10,
                    'sl_atr_multiplier': 2,
                    'tp_atr_multiplier': 3,
                    'atr_period': 14
                }
            }
        }
        
        simulator_config = {
            'montecarlo': {
                'module': 'strategy_module.utils.simulators',
                'class': 'MonteCarloSimulator',
                'default_params': {
                    'num_paths': 100,
                    'forecast_periods': 30,
                    'volatility_window': 20,
                    'drift_window': 20
                }
            }
        }
        
        # Write config files
        strategy_config_path = os.path.join(self.test_dir, 'config_strategies.json')
        simulator_config_path = os.path.join(self.test_dir, 'config_simulators.json')
        
        with open(strategy_config_path, 'w') as f:
            json.dump(strategy_config, f)
        with open(simulator_config_path, 'w') as f:
            json.dump(simulator_config, f)
        
        # Test loading configurations
        engine = create_forward_testing_engine(self.config)
        
        # Verify strategy creation
        strategy = engine._create_strategy()
        self.assertIsNotNone(strategy)
        
        # Verify simulator creation
        simulators = engine._create_simulators()
        self.assertEqual(len(simulators), 1)
        self.assertEqual(simulators[0].get_name(), 'MonteCarloSimulator')
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Completed test_config_file_loading in {execution_time:.2f} seconds - PASSED")

    def test_error_recovery(self):
        """Test error recovery and graceful degradation."""
        logger.info("Starting test_error_recovery")
        start_time = datetime.now()
        
        # Test with invalid strategy configuration
        os.environ['STRATEGY_CONFIG_PATH'] = 'nonexistent_path.json'
        with self.assertRaises(ValueError):  # Changed from FileNotFoundError to ValueError
            engine = create_forward_testing_engine(self.config)
            engine._create_strategy()
        
        # Test with invalid simulator configuration
        invalid_config = ForwardConfig(
            candle_data=self.sample_candle_data,
            simulators=[
                {
                    'type': 'invalid_simulator',
                    'parameters': {}
                }
            ],
            strategy_name='moving_average',
            strategy_params={}
        )
        with self.assertRaises(ValueError):
            engine = create_forward_testing_engine(invalid_config)
            engine.run()
        
        # Test with invalid candle data
        invalid_data_config = ForwardConfig(
            candle_data=None,
            simulators=[
                {
                    'type': 'monte_carlo',
                    'parameters': {
                        'num_paths': 100,
                        'forecast_periods': 30
                    }
                }
            ],
            strategy_name='moving_average',
            strategy_params={}
        )
        with self.assertRaises(ValueError):
            engine = create_forward_testing_engine(invalid_data_config)
            engine.run()
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Completed test_error_recovery in {execution_time:.2f} seconds - PASSED")

if __name__ == '__main__':
    unittest.main() 