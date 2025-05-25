"""
Strategy Tester module for the trading system.
Handles strategy implementation, backtesting, and performance analysis.
"""

import logging
import pandas as pd
import numpy as np
import os
from typing import Dict, Optional, Type, List, Union
from dataclasses import dataclass
from .backward_testing import BacktestingTradeEngine
from .forward_testing import ForwardTesting
from .statistical_testing import AdvancedStatisticalTesting
from .strategy_evaluation_metrics import StrategyEvaluationMetrics
from datetime import datetime
import json

# Setup logging
logger = logging.getLogger(__name__)

class StrategyTester:
    def __init__(self, 
                 ticker: str, 
                 timeframe: str, 
                 testing_period_months: int,
                 strategy: Type,
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
                 data_validation: bool = True,
                 # Input data
                 data: Optional[pd.DataFrame] = None,
                 # Strategy parameters
                 strategy_params: Optional[Dict] = None):
        """
        Initialize strategy tester with optimized parameters.
        
        Args:
            ticker: Trading symbol
            timeframe: Trading timeframe
            testing_period_months: Testing period in months
            strategy: Strategy class to test
            
            # Backtesting parameters
            forward_test: Whether to run forward testing
            trade_engine_type: Type of trade engine (Signals/SLTP)
            investing_amount: Initial account balance
            account_leverage: Account leverage
            volume: Trading volume
            commission: Commission per trade
            base_price: Base price column name
            
            # Backward testing specific parameters
            separate_close_signals: Whether to handle close signals separately
            parallel_opening: Whether to allow parallel position opening
            max_open_positions: Maximum number of open positions allowed
            position_sizing_method: Method for position sizing (fixed/risk_based/kelly)
            allow_shorting: Whether to allow short positions
            allow_partial_fills: Whether to allow partial order fills
            slippage_model: Model for slippage calculation
            slippage_value: Value for slippage calculation
            
            # Statistical testing parameters
            significance_level: Significance level for statistical tests
            
            # Forward testing parameters
            monte_carlo_simulations: Number of Monte Carlo simulations
            confidence_level: Confidence level for forward testing
            
            # Trade engine parameters
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            trailing_stop: Whether to use trailing stop
            trailing_stop_pct: Trailing stop percentage
            
            # Risk management parameters
            max_position_size: Maximum position size as fraction of portfolio
            max_drawdown: Maximum allowed drawdown
            risk_per_trade: Risk per trade as fraction of portfolio
            
            # Performance metrics parameters
            benchmark_symbol: Symbol for benchmark comparison
            risk_free_rate: Risk-free rate for calculations
            
            # Data validation parameters
            required_columns: List of required columns in data
            data_validation: Whether to validate input data
            
            # Input data
            data: Input price data
            
            # Strategy parameters
            strategy_params: Parameters for the strategy
        """
        logger.info("Initializing StrategyTester...")
        
        # Core parameters
        self.ticker = ticker
        self.timeframe = timeframe
        self.testing_period_months = testing_period_months
        self.strategy = strategy
        self.strategy_params = strategy_params or {}
        
        # Backtesting parameters
        self.volume = volume
        self.balance = investing_amount
        self.price = base_price
        self.leverage = account_leverage
        self.commission = commission
        self.forward_test = forward_test
        self.trade_engine_type = trade_engine_type
        
        # Backward testing specific parameters
        self.separate_close_signals = separate_close_signals
        self.parallel_opening = parallel_opening
        self.max_open_positions = max_open_positions
        self.position_sizing_method = position_sizing_method
        self.allow_shorting = allow_shorting
        self.allow_partial_fills = allow_partial_fills
        self.slippage_model = slippage_model
        self.slippage_value = slippage_value
        
        # Statistical testing parameters
        self.significance_level = significance_level
        
        # Forward testing parameters
        self.monte_carlo_simulations = monte_carlo_simulations
        self.confidence_level = confidence_level
        
        # Trade engine parameters
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop = trailing_stop
        self.trailing_stop_pct = trailing_stop_pct
        
        # Risk management parameters
        self.max_position_size = max_position_size
        self.max_drawdown = max_drawdown
        self.risk_per_trade = risk_per_trade
        
        # Performance metrics parameters
        self.benchmark_symbol = benchmark_symbol
        self.risk_free_rate = risk_free_rate
        
        # Data validation parameters
        self.required_columns = required_columns or ['time', self.price, 'Signal']
        if self.trade_engine_type == "SLTP":
            self.required_columns.extend(['SL', 'TP'])
        self.data_validation = data_validation
        
        # Initialize components
        logger.info("Initializing trade engine...")
        self.trade_engine = BacktestingTradeEngine(
            separate_close_signals=self.separate_close_signals,
            parallel_opening=self.parallel_opening
        )
        
        logger.info("Initializing statistical tester...")
        self.statistical_tester = AdvancedStatisticalTesting(
            significance_level=self.significance_level
        )
        
        # Validate and store input data
        if data is None:
            raise ValueError("Data must be provided to StrategyTester")
        if self.data_validation:
            self._validate_data(data)
        self.data = data
        
        # Initialize results containers
        logger.info("Initializing results containers...")
        self.trades = pd.DataFrame()
        self.metrics = None
        self.stats = None
        self.final_recommendation = None
        
        # Run backtest
        logger.info("Starting backtest...")
        self.run_backtest()
        
        # Run forward test if requested
        if self.forward_test:
            logger.info("Starting forward test...")
            self.run_forward_test()
            
        # Run statistical tests
        logger.info("Starting statistical tests...")
        self.run_statistical_tests()
        
        logger.info("StrategyTester initialization completed.")

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data structure."""
        missing_columns = [col for col in self.required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Input data missing required columns: {missing_columns}")

    def run_backtest(self) -> None:
        """Run backtest with optimized data processing."""
        logger.info("Running backtest...")
        
        # Prepare input data
        input_data = self._prepare_input_data()
        
        # Execute trades
        results = self._execute_trades(input_data)
        
        if results['trades']:
            self.trades = pd.DataFrame(results['trades'])
            logger.info(f"Trades DataFrame created with {len(self.trades)} trades")
            logger.info(f"Trades DataFrame columns: {self.trades.columns.tolist()}")
            
            self.metrics = self._calculate_metrics()
            logger.info(f"Metrics calculated: {len(self.metrics) if self.metrics is not None else 0} metrics")
            
            self.stats = self._calculate_statistics()
            logger.info(f"Statistics calculated: {len(self.stats) if self.stats is not None else 0} statistics")
            
            self.final_recommendation = self._generate_recommendation()
            logger.info(f"Final recommendation: {self.final_recommendation}")
            
            logger.info(f"Backtest completed. Total trades: {len(self.trades)}")
        else:
            logger.warning("No trades were executed during the backtest period")
            self._initialize_empty_results()

    def _prepare_input_data(self) -> pd.DataFrame:
        """Prepare input data for trade execution."""
        input_data = self.data[self.required_columns].copy()
        input_data.columns = ['time', 'Price', 'Signal'] + (['SL', 'TP'] if self.trade_engine_type == "SLTP" else [])
        return input_data

    def _execute_trades(self, input_data: pd.DataFrame) -> Dict:
        """Execute trades using the appropriate engine."""
        try:
            if self.trade_engine_type == "Signals":
                results = self.trade_engine.signal_based_trade_executor(
                    input_data, self.balance, self.leverage, self.volume, self.commission
                )
            else:
                results = self.trade_engine.sltp_based_trade_executor(
                    input_data, self.balance, self.leverage, self.volume, self.commission
                )
            
            # Ensure trades DataFrame has required columns
            if results['trades']:
                trades_df = pd.DataFrame(results['trades'])
                required_columns = ['time', 'Price', 'Signal', 'PnL', 'Balance']
                missing_columns = [col for col in required_columns if col not in trades_df.columns]
                
                if missing_columns:
                    logger.error(f"Missing required columns in trades: {missing_columns}")
                    logger.error(f"Available columns: {trades_df.columns.tolist()}")
                    raise ValueError(f"Trade execution results missing required columns: {missing_columns}")
                
                # Convert PnL to numeric if it's not already
                trades_df['PnL'] = pd.to_numeric(trades_df['PnL'], errors='coerce')
                
                # Update results with validated DataFrame
                results['trades'] = trades_df.to_dict('records')
            
            return results
            
        except Exception as e:
            logger.error(f"Error executing trades: {str(e)}")
            raise

    def _calculate_metrics(self) -> pd.DataFrame:
        """Calculate performance metrics using vectorized operations."""
        if self.trades.empty:
            return pd.DataFrame()
        
        try:
            # Initialize metrics calculator with correct parameters
            metrics_calculator = StrategyEvaluationMetrics(
                data=self.trades,
                balance=self.balance,
                ticker=self.ticker
            )
            
            # Calculate metrics
            metrics_calculator._calculate_evaluation_metrics()
            return metrics_calculator.metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            logger.error(f"Trades DataFrame columns: {self.trades.columns.tolist()}")
            raise

    def _calculate_statistics(self) -> pd.DataFrame:
        """Calculate trading statistics using vectorized operations."""
        if self.trades.empty:
            return pd.DataFrame()
            
        try:
            # Initialize metrics calculator with correct parameters
            metrics_calculator = StrategyEvaluationMetrics(
                data=self.trades,
                balance=self.balance,
                ticker=self.ticker
            )
            
            # Calculate statistics
            metrics_calculator._calculate_strategy_statistics()
            return metrics_calculator.stats
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {str(e)}")
            logger.error(f"Trades DataFrame columns: {self.trades.columns.tolist()}")
            raise

    def _generate_recommendation(self) -> str:
        """Generate trading recommendation based on metrics."""
        if self.stats is None or self.stats.empty:
            return "Insufficient data for recommendation"
            
        # Get key metrics
        win_rate = self.stats.loc[self.stats['Statistic'] == 'Win_Rate', 'Value'].values[0]
        profit_factor = self.stats.loc[self.stats['Statistic'] == 'Profit_Factor', 'Value'].values[0]
        sharpe_ratio = self.stats.loc[self.stats['Statistic'] == 'Sharpe_Ratio', 'Value'].values[0]
        
        # Generate recommendation
        if win_rate >= 50 and profit_factor >= 1.5 and sharpe_ratio >= 1:
            return "Strong Buy"
        elif win_rate >= 45 and profit_factor >= 1.2 and sharpe_ratio >= 0.5:
            return "Buy"
        elif win_rate >= 40 and profit_factor >= 1:
            return "Neutral"
        else:
            return "Sell"

    def _initialize_empty_results(self) -> None:
        """Initialize empty results when no trades are executed."""
        self.trades = pd.DataFrame()
        self.metrics = pd.DataFrame()
        self.stats = pd.DataFrame()
        self.final_recommendation = "No trades executed"

    def run_forward_test(self) -> None:
        """Run forward testing with Monte Carlo simulation."""
        logger.info("Running forward testing...")
        
        # Initialize forward testing with correct parameters
        forward_tester = ForwardTesting(
            data=self.data,
            base_price=self.price
        )
        
        # Generate future scenarios
        forward_tester.generate_future_prices()
        
        # Get scenario statistics
        self.forward_test_stats = forward_tester.get_all_scenarios_statistics()
        
        # Store forward testing results in a structured format
        scenario_names = ['Bullish', 'Neutral', 'Bearish']
        self.forward_test_results = {}
        
        # Initialize strategy with parameters
        strategy_instance = self.strategy(**self.strategy_params)
        
        for i, scenario_name in enumerate(scenario_names):
            try:
                scenario_data = forward_tester.future_data_frames[i].copy()
                
                # Ensure required columns are present
                required_columns = ['time', self.price]
                if not all(col in scenario_data.columns for col in required_columns):
                    logger.error(f"Missing required columns in scenario data: {required_columns}")
                    continue
                
                # Generate signals for the scenario
                signals = strategy_instance.generate_signals(scenario_data)
                if signals is None or signals.empty:
                    logger.warning(f"No signals generated for scenario {scenario_name}")
                    continue
                
                # Execute trades for the scenario
                if self.trade_engine_type == "Signals":
                    input_data = signals[['time', self.price, 'Signal']].copy()
                    input_data.columns = ['time', 'Price', 'Signal']
                    results = self.trade_engine.signal_based_trade_executor(
                        input_data, self.balance, self.leverage, self.volume, self.commission
                    )
                else:
                    input_data = signals[['time', self.price, 'Signal', 'SL', 'TP']].copy()
                    input_data.columns = ['time', 'Price', 'Signal', 'SL', 'TP']
                    results = self.trade_engine.sltp_based_trade_executor(
                        input_data, self.balance, self.leverage, self.volume, self.commission
                    )
                
                # Calculate metrics and statistics for the scenario
                if results['trades']:
                    trades_df = pd.DataFrame(results['trades'])
                    metrics_calculator = StrategyEvaluationMetrics(
                        data=trades_df,
                        balance=self.balance,
                        ticker=self.ticker
                    )
                    
                    self.forward_test_results[scenario_name] = {
                        'metrics': metrics_calculator._calculate_evaluation_metrics(),
                        'statistics': metrics_calculator._calculate_strategy_statistics(),
                        'trades': results['trades']
                    }
                else:
                    self.forward_test_results[scenario_name] = {
                        'metrics': pd.DataFrame(),
                        'statistics': pd.DataFrame(),
                        'trades': []
                    }
                    
            except Exception as e:
                logger.error(f"Error processing scenario {scenario_name}: {str(e)}")
                self.forward_test_results[scenario_name] = {
                    'metrics': pd.DataFrame(),
                    'statistics': pd.DataFrame(),
                    'trades': [],
                    'error': str(e)
                }
        
        # Plot scenarios
        try:
            forward_tester.plot_all_scenarios()
        except Exception as e:
            logger.error(f"Error plotting scenarios: {str(e)}")
        
        logger.info("Forward testing completed")

    def get_test_results(self, include_forward_test: bool = True) -> Dict:
        """Get test results in a structured format."""
        logger.info("Getting test results...")
        
        try:
            # Create results dictionary with safe access to attributes
            results = {
                'trades': self.trades.to_dict('records') if not self.trades.empty else [],
                'metrics': self.metrics.to_dict('records') if self.metrics is not None and not self.metrics.empty else [],
                'statistics': self.stats.to_dict('records') if self.stats is not None and not self.stats.empty else [],
                'recommendation': self.final_recommendation if hasattr(self, 'final_recommendation') else "No recommendation available",
                'statistical_tests': {}
            }
            
            # Add statistical test results if available
            if hasattr(self, 'ljung_box_results') and self.ljung_box_results is not None:
                try:
                    results['statistical_tests']['ljung_box_results'] = {
                        'test_statistic': float(self.ljung_box_results[0]),
                        'p_value': float(self.ljung_box_results[1]['lb_pvalue'][-1]),
                        'interpretation': 'Significant autocorrelation detected' if self.ljung_box_results[1]['lb_pvalue'][-1] < self.significance_level else 'No significant autocorrelation'
                    }
                except (KeyError, IndexError, TypeError) as e:
                    logger.warning(f"Error processing Ljung-Box results: {str(e)}")
            
            if hasattr(self, 'jarque_bera_results') and self.jarque_bera_results is not None:
                try:
                    results['statistical_tests']['jarque_bera_results'] = {
                        'test_statistic': float(self.jarque_bera_results[0]),
                        'p_value': float(self.jarque_bera_results[1]),
                        'interpretation': 'Returns are not normally distributed' if self.jarque_bera_results[1] < self.significance_level else 'Returns may be normally distributed'
                    }
                except (KeyError, IndexError, TypeError) as e:
                    logger.warning(f"Error processing Jarque-Bera results: {str(e)}")
            
            if hasattr(self, 'hurst_exponent') and self.hurst_exponent is not None:
                try:
                    results['statistical_tests']['hurst_exponent'] = {
                        'value': float(self.hurst_exponent),
                        'interpretation': 'Trending behavior' if self.hurst_exponent > 0.5 else 'Mean-reverting behavior' if self.hurst_exponent < 0.5 else 'Random walk behavior'
                    }
                except (KeyError, IndexError, TypeError) as e:
                    logger.warning(f"Error processing Hurst exponent: {str(e)}")
            
            if hasattr(self, 'auto_corr') and hasattr(self, 'cross_corr') and self.auto_corr is not None and self.cross_corr is not None:
                try:
                    results['statistical_tests']['correlation_analysis'] = {
                        'auto_corr_max': float(np.max(self.auto_corr)),
                        'auto_corr_lag': int(np.argmax(self.auto_corr)),
                        'cross_corr_max': float(np.max(self.cross_corr)),
                        'cross_corr_lag': int(np.argmax(self.cross_corr)),
                        'plot_file': os.path.join(os.path.dirname(__file__), '..', '..', '..', 'plots', f'correlations_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
                    }
                except (KeyError, IndexError, TypeError) as e:
                    logger.warning(f"Error processing correlation analysis: {str(e)}")
            
            # Add forward test results if available and requested
            if include_forward_test and hasattr(self, 'forward_test_results') and self.forward_test_results is not None:
                try:
                    results['forward_test'] = self.forward_test_results
                    logger.info("Forward test results included")
                except Exception as e:
                    logger.warning(f"Error processing forward test results: {str(e)}")
            
            # Log results structure
            logger.info("Test results structure:")
            logger.info(f"Keys in results: {list(results.keys())}")
            logger.info(f"Number of trades: {len(results['trades'])}")
            logger.info(f"Number of metrics: {len(results['metrics'])}")
            logger.info(f"Number of statistics: {len(results['statistics'])}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting test results: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error details: {str(e)}")
            # Return a minimal valid results structure
            return {
                'trades': [],
                'metrics': [],
                'statistics': [],
                'recommendation': "Error generating results",
                'statistical_tests': {}
            }

    def run_statistical_tests(self) -> Dict:
        """Run statistical tests on strategy performance and return results.
        
        Returns:
            Dict: Dictionary containing results of all statistical tests
        """
        logger.info("Running statistical tests...")
        
        results = {
            'ljung_box': {},
            'jarque_bera': {},
            'hurst_exponent': {},
            'correlation_analysis': {}
        }
        
        if self.trades.empty:
            logger.warning("No trades available for statistical testing")
            return results
            
        # Get returns series
        returns = self.trades['PnL'].values
        
        # Run Ljung-Box test for autocorrelation
        self.ljung_box_results = self.statistical_tester.ljung_box_test(returns)
        if self.ljung_box_results is not None:
            results['ljung_box'] = {
                'test_statistic': float(self.ljung_box_results[0]),
                'p_value': float(self.ljung_box_results[1]['lb_pvalue'][-1]),
                'interpretation': 'Significant autocorrelation detected' if self.ljung_box_results[1]['lb_pvalue'][-1] < self.significance_level else 'No significant autocorrelation'
            }
        
        # Run Jarque-Bera test for normality
        self.jarque_bera_results = self.statistical_tester.jarque_bera_test(returns)
        if self.jarque_bera_results is not None:
            results['jarque_bera'] = {
                'test_statistic': float(self.jarque_bera_results[0]),
                'p_value': float(self.jarque_bera_results[1]),
                'interpretation': 'Returns are not normally distributed' if self.jarque_bera_results[1] < self.significance_level else 'Returns may be normally distributed'
            }
        
        # Calculate Hurst exponent
        self.hurst_exponent = self.statistical_tester.hurst_exponent(returns)
        if self.hurst_exponent is not None:
            results['hurst_exponent'] = {
                'value': float(self.hurst_exponent),
                'interpretation': 'Trending behavior' if self.hurst_exponent > 0.5 else 'Mean-reverting behavior' if self.hurst_exponent < 0.5 else 'Random walk behavior'
            }
        
        # Calculate correlations
        self.auto_corr, self.cross_corr = self.statistical_tester.calculate_correlations(
            returns, self.data[self.price].values
        )
        if self.auto_corr is not None and self.cross_corr is not None:
            results['correlation_analysis'] = {
                'auto_corr_max': float(np.max(self.auto_corr)),
                'auto_corr_lag': int(np.argmax(self.auto_corr)),
                'cross_corr_max': float(np.max(self.cross_corr)),
                'cross_corr_lag': int(np.argmax(self.cross_corr)),
                'plot_file': os.path.join(os.path.dirname(__file__), '..', '..', '..', 'plots', f'correlations_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            }
        
        logger.info("Statistical testing completed")
        return results

    def store_test_results(self, output_dir: str = None) -> None:
        """Store test results in separate JSON files.
        
        Args:
            output_dir (str, optional): Directory to store the JSON files. 
                                      If None, uses a default directory.
        """
        logger.info("Storing test results...")
        
        try:
            # Create output directory if not provided
            if output_dir is None:
                output_dir = os.path.join(
                    os.path.dirname(__file__),
                    '..',
                    'results',
                    f'test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
                )
            
            # Create directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Get all test results
            results = self.get_test_results()
            stats_results = self.run_statistical_tests()
            
            # Store trades
            trades_file = os.path.join(output_dir, 'trades.json')
            with open(trades_file, 'w') as f:
                json.dump(results['trades'], f, indent=4)
            logger.info(f"Trades stored in {trades_file}")
            
            # Store metrics
            metrics_file = os.path.join(output_dir, 'metrics.json')
            with open(metrics_file, 'w') as f:
                json.dump(results['metrics'], f, indent=4)
            logger.info(f"Metrics stored in {metrics_file}")
            
            # Store statistics
            stats_file = os.path.join(output_dir, 'statistics.json')
            with open(stats_file, 'w') as f:
                json.dump(results['statistics'], f, indent=4)
            logger.info(f"Statistics stored in {stats_file}")
            
            # Store recommendation
            recommendation_file = os.path.join(output_dir, 'recommendation.json')
            with open(recommendation_file, 'w') as f:
                json.dump({'recommendation': results['recommendation']}, f, indent=4)
            logger.info(f"Recommendation stored in {recommendation_file}")
            
            # Store statistical test results
            stats_tests_file = os.path.join(output_dir, 'statistical_tests.json')
            with open(stats_tests_file, 'w') as f:
                json.dump(stats_results, f, indent=4)
            logger.info(f"Statistical test results stored in {stats_tests_file}")
            
            # Store configuration
            config_file = os.path.join(output_dir, 'config.json')
            config = {
                'ticker': self.ticker,
                'timeframe': self.timeframe,
                'testing_period_months': self.testing_period_months,
                'strategy_params': self.strategy_params,
                'backtesting_params': {
                    'forward_test': self.forward_test,
                    'trade_engine_type': self.trade_engine_type,
                    'investing_amount': self.balance,
                    'account_leverage': self.leverage,
                    'volume': self.volume,
                    'commission': self.commission,
                    'base_price': self.price
                },
                'backward_testing_params': {
                    'separate_close_signals': self.separate_close_signals,
                    'parallel_opening': self.parallel_opening,
                    'max_open_positions': self.max_open_positions,
                    'position_sizing_method': self.position_sizing_method,
                    'allow_shorting': self.allow_shorting,
                    'allow_partial_fills': self.allow_partial_fills,
                    'slippage_model': self.slippage_model,
                    'slippage_value': self.slippage_value
                },
                'statistical_params': {
                    'significance_level': self.significance_level,
                    'monte_carlo_simulations': self.monte_carlo_simulations,
                    'confidence_level': self.confidence_level
                },
                'trade_engine_params': {
                    'stop_loss_pct': self.stop_loss_pct,
                    'take_profit_pct': self.take_profit_pct,
                    'trailing_stop': self.trailing_stop,
                    'trailing_stop_pct': self.trailing_stop_pct
                },
                'risk_management_params': {
                    'max_position_size': self.max_position_size,
                    'max_drawdown': self.max_drawdown,
                    'risk_per_trade': self.risk_per_trade
                },
                'performance_metrics_params': {
                    'benchmark_symbol': self.benchmark_symbol,
                    'risk_free_rate': self.risk_free_rate
                },
                'data_validation_params': {
                    'required_columns': self.required_columns,
                    'data_validation': self.data_validation
                },
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=4)
            logger.info(f"Configuration stored in {config_file}")
            
            logger.info(f"All test results stored in {output_dir}")
            
        except Exception as e:
            logger.error(f"Error storing test results: {str(e)}")
            raise

