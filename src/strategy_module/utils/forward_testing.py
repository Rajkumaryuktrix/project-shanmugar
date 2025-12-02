import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Protocol, Union, TYPE_CHECKING
import logging
import matplotlib.pyplot as plt
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from .backward_testing import BacktestingEngine, CandleData
from numba import jit

if TYPE_CHECKING:
    from .simulators import BaseSimulator, MonteCarloSimulator, MonteCarloConfig

logger = logging.getLogger(__name__)

@dataclass
class SimulatorConfig:
    """Base configuration for simulators."""
    name: str
    enabled: bool = True
    params: Dict[str, Any] = None

@dataclass
class ForwardConfig:
    """Configuration for forward testing."""
    candle_data: CandleData
    backtest_engine: BacktestingEngine
    simulators: List[Dict[str, Any]]
    strategy_name: str = 'moving_average'
    strategy_params: Dict[str, Any] = None
    confidence_level: float = 0.95
    significance_level: float = 0.05
    risk_per_trade: float = 0.02
    max_position_size: float = 100000
    max_drawdown: float = 0.2
    save_results: bool = True
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if not isinstance(self.candle_data, CandleData):
            raise ValueError("Candle data must be a CandleData object")
            
        if not isinstance(self.backtest_engine, BacktestingEngine):
            raise ValueError("Backtest engine must be a BacktestingEngine object")
            
        if not isinstance(self.simulators, list):
            raise ValueError("Simulators must be a list")
            
        if not self.simulators:
            raise ValueError("At least one simulator must be specified")
            
        # Validate simulator configurations
        valid_simulator_types = {'monte_carlo', 'brownian_motion'}
        for sim_config in self.simulators:
            if not isinstance(sim_config, dict):
                raise ValueError("Simulator configuration must be a dictionary")
                
            sim_type = sim_config.get('type')
            if not sim_type:
                raise ValueError("Simulator type not specified")
                
            if sim_type not in valid_simulator_types:
                raise ValueError(f"Invalid simulator type: {sim_type}")
                
            if not isinstance(sim_config.get('enabled', True), bool):
                raise ValueError("Simulator enabled flag must be a boolean")
                
            if not isinstance(sim_config.get('parameters', {}), dict):
                raise ValueError("Simulator parameters must be a dictionary")
        
        # Validate strategy name
        valid_strategies = {'moving_average', 'rsi', 'time_based'}
        if self.strategy_name not in valid_strategies:
            raise ValueError(f"Invalid strategy name: {self.strategy_name}")
            
        # Validate confidence level
        if not 0 < self.confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1")
            
        # Validate significance level
        if not 0 < self.significance_level < 1:
            raise ValueError("Significance level must be between 0 and 1")
            
        # Validate risk per trade
        if not 0 < self.risk_per_trade <= 1:
            raise ValueError("Risk per trade must be between 0 and 1")
            
        # Validate max position size
        if self.max_position_size <= 0:
            raise ValueError("Max position size must be greater than 0")
            
        # Validate max drawdown
        if not 0 < self.max_drawdown <= 1:
            raise ValueError("Max drawdown must be between 0 and 1")

        # Validate save_results
        if not isinstance(self.save_results, bool):
            raise ValueError("Save results must be a boolean")

@dataclass
class ForwardResult:
    """Results from forward testing."""
    # Simulation results
    scenarios: Dict[str, Dict[str, Any]]
    summary: Dict[str, Any]
    
    # Performance metrics
    metrics: Dict[str, float]
    
    # Configuration used
    config: ForwardConfig
    
    # Metadata
    timestamp: str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary format."""
        return {
            'scenarios': self.scenarios,
            'summary': self.summary,
            'metrics': self.metrics,
            'config': {
                'simulators': [
                    {
                        'type': sim.get('type'),
                        'enabled': sim.get('enabled', True),
                        'parameters': sim.get('parameters', {})
                    }
                    for sim in self.config.simulators
                ],
                'confidence_level': self.config.confidence_level,
                'significance_level': self.config.significance_level,
                'max_drawdown': self.config.max_drawdown,
                'risk_per_trade': self.config.risk_per_trade,
                'max_position_size': self.config.max_position_size
            },
            'timestamp': self.timestamp
        }

class BaseSimulator(ABC):
    """Abstract base class for all simulators."""
    
    def __init__(self, config: ForwardConfig):
        self.config = config
        self.validate_config()
    
    def validate_config(self) -> None:
        """Validate simulator-specific configuration."""
        if not isinstance(self.config, ForwardConfig):
            raise ValueError("Invalid configuration type. Expected ForwardConfig")
        
        if self.config.candle_data is None:
            raise ValueError("CandleData cannot be None")
            
        if len(self.config.candle_data.close) < 20:  # Minimum required for volatility calculation
            raise ValueError("Insufficient data for simulation. Need at least 20 periods")
            
        # Validate data consistency
        data_length = len(self.config.candle_data.close)
        if not all(len(arr) == data_length for arr in [
            self.config.candle_data.open,
            self.config.candle_data.high,
            self.config.candle_data.low,
            self.config.candle_data.volume,
            self.config.candle_data.timestamp
        ]):
            raise ValueError("Inconsistent data lengths in CandleData")
    
    def generate_paths(self) -> Dict[str, CandleData]:
        """Generate simulation paths."""
        raise NotImplementedError("Subclasses must implement generate_paths()")
    
    def run_simulation(self) -> Dict[str, Any]:
        """Run the simulation."""
        raise NotImplementedError("Subclasses must implement run_simulation()")
    
    def get_name(self) -> str:
        """Get simulator name."""
        return self.__class__.__name__

class ForwardTestingEngine:
    """Main engine for forward testing that orchestrates simulation execution."""
    
    def __init__(self, config: ForwardConfig):
        """
        Initialize forward testing engine.
        
        Args:
            config: Forward testing configuration
        """
        self.config = config
        self.config.validate()
        self.simulators: List['BaseSimulator'] = []
        self.results: Optional[Dict] = None
        
    def _create_simulators(self) -> List[Any]:
        """Create simulator instances based on configuration."""
        simulators = []
        for sim_config in self.config.simulators:
            try:
                # Validate simulator configuration
                if not isinstance(sim_config, dict):
                    logger.error(f"Invalid simulator configuration: {sim_config}")
                    continue
                
                # Get simulator type and parameters
                sim_type = sim_config.get('type')
                enabled = sim_config.get('enabled', True)
                params = sim_config.get('parameters', {})
                
                if not sim_type:
                    logger.error("Simulator type not specified")
                    continue
                
                if not enabled:
                    logger.info(f"Skipping disabled simulator: {sim_type}")
                    continue
                
                # Create simulator instance
                if sim_type == 'monte_carlo':
                    from .simulators import MonteCarloSimulator, MonteCarloConfig
                    mc_config = MonteCarloConfig(**params)
                    simulator = MonteCarloSimulator(self.config, mc_config)
                elif sim_type == 'brownian_motion':
                    from .simulators import BrownianMotionSimulator, BrownianMotionConfig
                    bm_config = BrownianMotionConfig(**params)
                    simulator = BrownianMotionSimulator(self.config, bm_config)
                else:
                    logger.error(f"Unknown simulator type: {sim_type}")
                    continue
                
                simulators.append(simulator)
                logger.info(f"Created {sim_type} simulator with parameters: {params}")
                
            except Exception as e:
                logger.error(f"Error creating simulator: {str(e)}")
                continue
        
        if not simulators:
            raise ValueError("No valid simulators could be created")
            
        return simulators
        
    def _generate_paths(self) -> Dict[str, CandleData]:
        """Generate simulation paths using selected simulators."""
        paths = {}
        for simulator in self.simulators:
            try:
                simulator_paths = simulator.generate_paths()
                paths.update(simulator_paths)
            except Exception as e:
                logger.error(f"Error generating paths for {simulator.get_name()}: {str(e)}")
                raise
        return paths
        
    def _run_backtests(self, paths: Dict[str, CandleData]) -> Dict[str, Dict]:
        """Run backtests on generated paths."""
        results = {}
        strategy = self._create_strategy()
        
        for path_name, path_data in paths.items():
            try:
                # Prepare data for strategy
                data = {
                    'time': path_data.timestamp,
                    'open': path_data.open,
                    'high': path_data.high,
                    'low': path_data.low,
                    'close': path_data.close,
                    'volume': path_data.volume
                }
                
                # Generate signals
                signals = strategy.generate_signals(data)
                
                # Update CandleData with signals
                path_data.signal = signals['signal']
                path_data.sl = signals['sl']
                path_data.tp = signals['tp']
                
                # Update backtesting engine data
                self.config.backtest_engine.data = path_data
                
                # Run backtest
                backtest_result = self.config.backtest_engine.run_backtest()
                
                # Calculate additional metrics
                total_trades = backtest_result.get('total_trades', 0)
                winning_trades = backtest_result.get('winning_trades', 0)
                losing_trades = backtest_result.get('losing_trades', 0)
                total_profit = backtest_result.get('total_profit', 0)
                max_drawdown = backtest_result.get('max_drawdown', 0)
                
                # Store results with metadata
                results[path_name] = {
                    'success': True,
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'total_profit': total_profit,
                    'max_drawdown': max_drawdown,
                    'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
                    'sharpe_ratio': backtest_result.get('sharpe_ratio', 0),
                    'strategy_name': self.config.strategy_name,
                    'strategy_params': self.config.strategy_params,
                    'signals': {
                        'total_signals': len(signals['signal']),
                        'buy_signals': int(np.sum(signals['signal'] == 1)),
                        'sell_signals': int(np.sum(signals['signal'] == -1)),
                        'long_signals': int(np.sum(signals['signal'] == 2)),
                        'short_signals': int(np.sum(signals['signal'] == -2))
                    },
                    'trades': backtest_result.get('trades', []),
                    'equity_curve': backtest_result.get('equity_curve', []),
                    'drawdown_curve': backtest_result.get('drawdown_curve', [])
                }
                
            except Exception as e:
                logger.error(f"Error processing path {path_name}: {str(e)}")
                results[path_name] = {
                    'success': False,
                    'error': str(e)
                }
                
        return results

    def _create_strategy(self) -> Any:
        """Create strategy instance based on configuration."""
        try:
            # Load strategy configuration from JSON file
            config_path = os.path.join(os.path.dirname(__file__), '..', 'strategies', 'config', 'config_strategies.json')
            with open(config_path, 'r') as f:
                strategy_modules = json.load(f)
            
            # Get strategy configuration
            strategy_config = strategy_modules.get(self.config.strategy_name.lower())
            if not strategy_config:
                raise ValueError(f"Unknown strategy type: {self.config.strategy_name}")
            
            # Import strategy module
            module_path = strategy_config['module']
            class_name = strategy_config['class']
            
            # Import the module
            try:
                module = __import__(module_path, fromlist=[class_name])
                strategy_class = getattr(module, class_name)
            except (ImportError, AttributeError) as e:
                logger.error(f"Error importing strategy {self.config.strategy_name}: {str(e)}")
                raise
            
            # Merge default parameters with user-provided parameters
            params = strategy_config['default_params'].copy()
            if self.config.strategy_params:
                params.update(self.config.strategy_params)
            
            # Create strategy instance with merged parameters
            return strategy_class(**params)
            
        except Exception as e:
            logger.error(f"Error creating strategy {self.config.strategy_name}: {str(e)}")
            raise

    def run(self) -> Dict[str, Any]:
        """Run forward testing process."""
        try:
            # Initialize simulators
            self.simulators = self._create_simulators()
            if not self.simulators:
                raise ValueError("No valid simulators could be initialized")

            # Generate simulation paths
            paths = self._generate_paths()
            if not paths:
                raise ValueError("No simulation paths were generated")

            # Run backtests on generated paths
            backtest_results = self._run_backtests(paths)
            if not backtest_results:
                raise ValueError("No backtest results were generated")

            # Calculate summary statistics
            summary = self._calculate_summary(backtest_results)

            # Create result object
            result = ForwardResult(
                scenarios=backtest_results,
                summary=summary,
                metrics=self._calculate_metrics(backtest_results),
                config=self.config
            )

            # Save results if configured
            if self.config.save_results:
                self._save_results(result.to_dict(), self.config.output_dir)

            return result.to_dict()

        except Exception as e:
            logger.error(f"Error in forward testing: {str(e)}")
            raise

    def _calculate_summary(self, backtest_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Calculate summary statistics from backtest results."""
        scenario_names = list(backtest_results.keys())
        successful_scenarios = [name for name, result in backtest_results.items() if result.get('success', False)]
        
        summary = {
            'total_scenarios': len(backtest_results),
            'successful_scenarios': len(successful_scenarios),
            'scenario_names': scenario_names,
            'successful_scenario_names': successful_scenarios,
            'total_trades': sum(r.get('total_trades', 0) for r in backtest_results.values()),
            'winning_trades': sum(r.get('winning_trades', 0) for r in backtest_results.values()),
            'losing_trades': sum(r.get('losing_trades', 0) for r in backtest_results.values()),
            'total_profit': sum(r.get('total_profit', 0) for r in backtest_results.values()),
            'max_drawdown': max(r.get('max_drawdown', 0) for r in backtest_results.values()),
            'sharpe_ratio': np.mean([r.get('sharpe_ratio', 0) for r in backtest_results.values()]),
            'win_rate': np.mean([r.get('win_rate', 0) for r in backtest_results.values()]),
            'avg_trades_per_scenario': np.mean([r.get('total_trades', 0) for r in backtest_results.values()]),
            'avg_profit_per_scenario': np.mean([r.get('total_profit', 0) for r in backtest_results.values()]),
            'avg_drawdown_per_scenario': np.mean([r.get('max_drawdown', 0) for r in backtest_results.values()]),
            'scenario_success_rate': len(successful_scenarios) / len(backtest_results) if backtest_results else 0
        }
        return summary

    def _calculate_metrics(self, backtest_results: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate performance metrics from backtest results."""
        metrics = {
            'avg_profit': np.mean([r.get('total_profit', 0) for r in backtest_results.values()]),
            'profit_std': np.std([r.get('total_profit', 0) for r in backtest_results.values()]),
            'avg_drawdown': np.mean([r.get('max_drawdown', 0) for r in backtest_results.values()]),
            'avg_sharpe': np.mean([r.get('sharpe_ratio', 0) for r in backtest_results.values()]),
            'avg_win_rate': np.mean([r.get('win_rate', 0) for r in backtest_results.values()]),
            'avg_trades': np.mean([r.get('total_trades', 0) for r in backtest_results.values()])
        }
        return metrics

    def _save_results(self, results: Dict[str, Any], output_dir: str) -> None:
        """Save results to a JSON file."""
        try:
            # Convert numpy types to Python native types
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                return obj

            results = convert_numpy_types(results)
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"forward_test_results_{timestamp}.json"
            filepath = os.path.join(output_dir, filename)
            
            # Save results
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=4)
            
            logger.info(f"Results saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")

def create_forward_testing_engine(config: ForwardConfig) -> ForwardTestingEngine:
    """
    Factory function to create forward testing engine.
    
    Args:
        config: Forward testing configuration
        
    Returns:
        Instance of forward testing engine
    """
    return ForwardTestingEngine(config)

