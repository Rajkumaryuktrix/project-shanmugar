import logging
from typing import Dict, List, Optional, Type, Union
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import json

from src.broker_module.upstox.data.CandleData import UpstoxHistoricalData
from src.strategy_module.utils.strategy_tester import StrategyTester

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StrategyEngine:
    """
    Core engine for strategy testing and optimization.
    Integrates data loading, strategy execution, and result analysis.
    """
    
    def __init__(self):
        """Initialize the strategy engine with required components."""
        self.candle_data = UpstoxHistoricalData()
        self.available_strategies = self._load_available_strategies()
        self.current_config = None
        self.current_results = None
        self.current_optimization_results = None
        
    def _load_available_strategies(self) -> Dict[str, Dict]:
        """Load all available strategies from the strategies folder."""
        strategies = {}
        strategies_dir = os.path.join(os.path.dirname(__file__), 'strategies')
        
        # Get all .py files in the strategies directory
        strategy_files = [f for f in os.listdir(strategies_dir) if f.endswith('_strategy.py')]
        
        for strategy_file in strategy_files:
            try:
                # Get module name without .py extension
                module_name = strategy_file[:-3]
                # Import the strategy module
                module = __import__(f'src.strategy_module.strategies.{module_name}', fromlist=[''])
                
                # Find the strategy class in the module
                for name, obj in module.__dict__.items():
                    if isinstance(obj, type) and name.endswith('Strategy'):
                        # Get strategy parameters from class docstring or __init__
                        strategy_name = module_name.replace('_strategy', '')
                        strategies[strategy_name] = {
                            'class': obj,
                            'module': module,
                            'params': self._get_strategy_parameters(obj)
                        }
                        break
            except Exception as e:
                logger.error(f"Error loading strategy {strategy_file}: {str(e)}")
                continue
        
        return strategies
    
    def _get_strategy_parameters(self, strategy_class: Type) -> Dict:
        """Extract strategy parameters from class docstring or __init__."""
        params = {}
        try:
            # Get parameters from class docstring
            if strategy_class.__doc__:
                doc_lines = strategy_class.__doc__.split('\n')
                for line in doc_lines:
                    if ':' in line:
                        param_name, param_desc = line.split(':', 1)
                        param_name = param_name.strip()
                        if param_name and not param_name.startswith(' '):
                            params[param_name] = {
                                'description': param_desc.strip(),
                                'type': 'str',  # Default type
                                'default': None
                            }
            
            # Get parameters from __init__
            if hasattr(strategy_class, '__init__'):
                import inspect
                sig = inspect.signature(strategy_class.__init__)
                for name, param in sig.parameters.items():
                    if name != 'self':
                        param_type = str(param.annotation) if param.annotation != inspect.Parameter.empty else 'str'
                        default = param.default if param.default != inspect.Parameter.empty else None
                        
                        if name in params:
                            params[name].update({
                                'type': param_type,
                                'default': default
                            })
                        else:
                            params[name] = {
                                'description': f'Parameter {name}',
                                'type': param_type,
                                'default': default
                            }
        except Exception as e:
            logger.error(f"Error extracting strategy parameters: {str(e)}")
        
        return params
    
    def load_data(self, 
                 instrument_key: str,
                 unit: str,
                 interval: int,
                 days: int,
                 data_type: str = 'historical',
                 to_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load data for strategy testing.
        
        Args:
            instrument_key (str): Instrument key for the symbol
            unit (str): Time unit (minutes, hours, days)
            interval (int): Time interval
            days (int): Number of days of historical data (only for historical data type)
            data_type (str): Type of data to load ('historical' or 'intraday')
            to_date (str, optional): End date in YYYY-MM-DD format (only for historical data type)
            
        Returns:
            pd.DataFrame: Data with OHLCV columns
            
        Raises:
            ValueError: If data_type is invalid or required parameters are missing
        """
        try:
            if data_type not in ['historical', 'intraday']:
                raise ValueError("data_type must be either 'historical' or 'intraday'")
            
            if data_type == 'historical':
                if not to_date:
                    to_date = datetime.now().strftime('%Y-%m-%d')
                
                from_date = (datetime.strptime(to_date, '%Y-%m-%d') - timedelta(days=days)).strftime('%Y-%m-%d')
                
                data = self.candle_data.get_historical_candles(
                    instrument_key=instrument_key,
                    unit=unit,
                    interval=interval,
                    to_date=to_date,
                    from_date=from_date
                )
            else:  # intraday
                data = self.candle_data.get_intraday_candles(
                    instrument_key=instrument_key,
                    unit=unit,
                    interval=interval
                )
            
            if data.empty:
                raise ValueError("No data available for the specified parameters")
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def run_strategy(self,
                    data: pd.DataFrame,
                    strategy_name: str,
                    strategy_params: Dict,
                    tester_params: Dict,
                    # Backtesting parameters
                    forward_test: bool = True,
                    trade_engine_type: str = "Signals",
                    investing_amount: float = 1000,
                    account_leverage: float = 500,
                    volume: float = 0.01,
                    commission: float = 7,
                    base_price: str = "close",
                    # Backward testing specific parameters
                    separate_close_signals: bool = False,
                    parallel_opening: bool = False,
                    max_open_positions: int = 1,
                    position_sizing_method: str = "fixed",  # fixed, risk_based, kelly
                    allow_shorting: bool = True,
                    allow_partial_fills: bool = False,
                    slippage_model: str = "none",  # none, fixed, percentage, random
                    slippage_value: float = 0.0,
                    # Statistical testing parameters
                    significance_level: float = 0.05,
                    # Forward testing parameters
                    monte_carlo_simulations: int = 1000,
                    confidence_level: float = 0.95,
                    # Trade engine parameters
                    stop_loss_pct: float = 2.0,
                    take_profit_pct: float = 4.0,
                    trailing_stop: bool = False,
                    trailing_stop_pct: float = 1.0,
                    # Risk management parameters
                    max_position_size: float = 0.1,
                    max_drawdown: float = 0.2,
                    risk_per_trade: float = 0.02,
                    # Performance metrics parameters
                    benchmark_symbol: Optional[str] = None,
                    risk_free_rate: float = 0.05,
                    # Data validation parameters
                    required_columns: Optional[List[str]] = None,
                    data_validation: bool = True) -> Dict:
        """
        Run a strategy test with the given parameters.
        
        Args:
            data (pd.DataFrame): Historical data
            strategy_name (str): Name of the strategy to run
            strategy_params (Dict): Strategy parameters
            tester_params (Dict): Basic tester parameters (ticker, timeframe, testing_period_months)
            
            # Backtesting parameters
            forward_test (bool): Whether to run forward testing
            trade_engine_type (str): Type of trade engine (Signals/SLTP)
            investing_amount (float): Initial account balance
            account_leverage (float): Account leverage
            volume (float): Trading volume
            commission (float): Commission per trade
            base_price (str): Base price column name
            
            # Backward testing specific parameters
            separate_close_signals (bool): Whether to handle close signals separately
            parallel_opening (bool): Whether to allow parallel position opening
            max_open_positions (int): Maximum number of open positions allowed
            position_sizing_method (str): Method for position sizing (fixed/risk_based/kelly)
            allow_shorting (bool): Whether to allow short positions
            allow_partial_fills (bool): Whether to allow partial order fills
            slippage_model (str): Model for slippage calculation
            slippage_value (float): Value for slippage calculation
            
            # Statistical testing parameters
            significance_level (float): Significance level for statistical tests
            
            # Forward testing parameters
            monte_carlo_simulations (int): Number of Monte Carlo simulations
            confidence_level (float): Confidence level for forward testing
            
            # Trade engine parameters
            stop_loss_pct (float): Stop loss percentage
            take_profit_pct (float): Take profit percentage
            trailing_stop (bool): Whether to use trailing stop
            trailing_stop_pct (float): Trailing stop percentage
            
            # Risk management parameters
            max_position_size (float): Maximum position size as fraction of portfolio
            max_drawdown (float): Maximum allowed drawdown
            risk_per_trade (float): Risk per trade as fraction of portfolio
            
            # Performance metrics parameters
            benchmark_symbol (str, optional): Symbol for benchmark comparison
            risk_free_rate (float): Risk-free rate for calculations
            
            # Data validation parameters
            required_columns (List[str], optional): List of required columns in data
            data_validation (bool): Whether to validate input data
            
        Returns:
            Dict: Test results including trades, metrics, and statistics
        """
        try:
            if strategy_name not in self.available_strategies:
                raise ValueError(f"Strategy {strategy_name} not found")
            
            # Initialize strategy
            strategy_class = self.available_strategies[strategy_name]['class']
            strategy = strategy_class(**strategy_params)
            
            # Generate signals
            data_with_signals = strategy.generate_signals(data)
            if data_with_signals.empty:
                raise ValueError("No signals generated")
            
            # Initialize tester with all parameters
            tester = StrategyTester(
                ticker=tester_params.get('ticker', ''),
                timeframe=tester_params.get('timeframe', ''),
                testing_period_months=tester_params.get('testing_period_months', 3),
                strategy=strategy,
                strategy_params=strategy_params,
                # Backtesting parameters
                forward_test=forward_test,
                trade_engine_type=trade_engine_type,
                investing_amount=investing_amount,
                account_leverage=account_leverage,
                volume=volume,
                commission=commission,
                base_price=base_price,
                # Backward testing specific parameters
                separate_close_signals=separate_close_signals,
                parallel_opening=parallel_opening,
                max_open_positions=max_open_positions,
                position_sizing_method=position_sizing_method,
                allow_shorting=allow_shorting,
                allow_partial_fills=allow_partial_fills,
                slippage_model=slippage_model,
                slippage_value=slippage_value,
                # Statistical testing parameters
                significance_level=significance_level,
                # Forward testing parameters
                monte_carlo_simulations=monte_carlo_simulations,
                confidence_level=confidence_level,
                # Trade engine parameters
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                trailing_stop=trailing_stop,
                trailing_stop_pct=trailing_stop_pct,
                # Risk management parameters
                max_position_size=max_position_size,
                max_drawdown=max_drawdown,
                risk_per_trade=risk_per_trade,
                # Performance metrics parameters
                benchmark_symbol=benchmark_symbol,
                risk_free_rate=risk_free_rate,
                # Data validation parameters
                required_columns=required_columns,
                data_validation=data_validation,
                # Input data
                data=data_with_signals
            )
            
            # Run test and get results
            results = tester.get_test_results()
            
            # Save results
            self.current_results = results
            self.current_config = {
                'strategy_name': strategy_name,
                'strategy_params': strategy_params,
                'tester_params': tester_params,
                'backtesting_params': {
                    'forward_test': forward_test,
                    'trade_engine_type': trade_engine_type,
                    'investing_amount': investing_amount,
                    'account_leverage': account_leverage,
                    'volume': volume,
                    'commission': commission,
                    'base_price': base_price
                },
                'backward_testing_params': {
                    'separate_close_signals': separate_close_signals,
                    'parallel_opening': parallel_opening,
                    'max_open_positions': max_open_positions,
                    'position_sizing_method': position_sizing_method,
                    'allow_shorting': allow_shorting,
                    'allow_partial_fills': allow_partial_fills,
                    'slippage_model': slippage_model,
                    'slippage_value': slippage_value
                },
                'statistical_params': {
                    'significance_level': significance_level,
                    'monte_carlo_simulations': monte_carlo_simulations,
                    'confidence_level': confidence_level
                },
                'trade_engine_params': {
                    'stop_loss_pct': stop_loss_pct,
                    'take_profit_pct': take_profit_pct,
                    'trailing_stop': trailing_stop,
                    'trailing_stop_pct': trailing_stop_pct
                },
                'risk_management_params': {
                    'max_position_size': max_position_size,
                    'max_drawdown': max_drawdown,
                    'risk_per_trade': risk_per_trade
                },
                'performance_metrics_params': {
                    'benchmark_symbol': benchmark_symbol,
                    'risk_free_rate': risk_free_rate
                },
                'data_validation_params': {
                    'required_columns': required_columns,
                    'data_validation': data_validation
                },
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Store results in files
            output_dir = os.path.join(
                os.path.dirname(__file__),
                'results',
                f'backtest_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            )
            os.makedirs(output_dir, exist_ok=True)
            
            # Save results to JSON files
            with open(os.path.join(output_dir, 'backtest_results.json'), 'w') as f:
                json.dump(results, f, indent=4)
            
            # Save configuration
            with open(os.path.join(output_dir, 'config.json'), 'w') as f:
                json.dump(self.current_config, f, indent=4)
            
            logger.info(f"Results saved to {output_dir}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error running strategy: {str(e)}")
            raise
    
    def optimize_strategy(self,
                         data: pd.DataFrame,
                         strategy_name: str,
                         param_ranges: Dict[str, List],
                         optimization_params: Dict) -> Dict:
        """
        Optimize strategy parameters using specified optimization method.
        
        Args:
            data (pd.DataFrame): Historical data
            strategy_name (str): Name of the strategy to optimize
            param_ranges (Dict[str, List]): Parameter ranges for optimization
            optimization_params (Dict): Optimization parameters including:
                - method (str): Optimization method ('grid_search' or 'differential_evolution')
                - criterion (str): Optimization criterion
                - max_combinations (int): Maximum number of parameter combinations to test
                - trade_engine_type (str): Type of trade engine
                - balance (float): Initial account balance
                - leverage (float): Account leverage
                - volume (float): Trading volume
                - commission (float): Commission per trade
                
        Returns:
            Dict: Optimization results including:
                - optimization_results: All parameter combinations and their results
                - best_parameters: Best parameter combination found
                - best_result_metrics: Detailed metrics for best parameters
                - config: Configuration used for optimization
        """
        try:
            if strategy_name not in self.available_strategies:
                raise ValueError(f"Strategy {strategy_name} not found")
            
            # Initialize strategy
            strategy_class = self.available_strategies[strategy_name]['class']
            
            # Convert param_ranges to parameter_bounds format
            parameter_bounds = []
            for param_name, param_range in param_ranges.items():
                if isinstance(param_range, (list, tuple)) and len(param_range) >= 2:
                    parameter_bounds.append((param_range[0], param_range[-1]))
                else:
                    raise ValueError(f"Invalid parameter range for {param_name}")
            
            # Initialize optimizer with all parameters
            from src.strategy_module.utils.strategy_optimizer import StrategyOptimizer
            optimizer = StrategyOptimizer(
                ticker=optimization_params.get('ticker', ''),
                strategy_class=strategy_class,
                parameter_bounds=parameter_bounds,
                data=data,
                criterion=optimization_params.get('criterion', 'Balance_Max'),
                timeframe=optimization_params.get('timeframe', '5minute'),
                trade_engine_type=optimization_params.get('trade_engine_type', 'Signals'),
                optimizing_period=optimization_params.get('optimizing_period', 3),
                balance=optimization_params.get('balance', 1000),
                leverage=optimization_params.get('leverage', 500),
                volume=optimization_params.get('volume', 1),
                commission=optimization_params.get('commission', 7),
                optimization_method=optimization_params.get('method', 'grid_search'),
                max_combinations=optimization_params.get('max_combinations', 1000)
            )
            
            # Run optimization
            optimizer.optimize_strategy()
            
            # Store results
            results = {
                'optimization_results': optimizer.get_optimization_results().to_dict('records'),
                'best_parameters': optimizer.get_best_parameters(),
                'best_result_metrics': optimizer.get_optimization_results().iloc[0].to_dict() if not optimizer.get_optimization_results().empty else None,
                'config': {
                    'strategy_name': strategy_name,
                    'param_ranges': param_ranges,
                    'optimization_params': optimization_params,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            }
            
            # Store results in files
            output_dir = os.path.join(
                os.path.dirname(__file__),
                'results',
                f'optimization_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            )
            optimizer.store_optimization_results(output_dir)
            
            # Update current state
            self.current_optimization_results = results
            self.current_config = results['config']
            
            logger.info(f"Optimization results saved to {output_dir}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error optimizing strategy: {str(e)}")
            raise
    
    def get_available_strategies(self) -> Dict[str, Dict]:
        """Get list of available strategies with their parameters."""
        return self.available_strategies
    
    def get_current_results(self) -> Optional[Dict]:
        """Get the results of the most recent strategy test."""
        return self.current_results
    
    def get_current_optimization_results(self) -> Optional[Dict]:
        """Get the results of the most recent strategy optimization."""
        return self.current_optimization_results
    
    def get_current_config(self) -> Optional[Dict]:
        """Get the configuration of the most recent test/optimization."""
        return self.current_config 