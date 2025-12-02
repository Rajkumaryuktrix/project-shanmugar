"""
Strategy Tester module for the trading system.
Handles strategy implementation, backtesting, and performance analysis.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional, Type, List, Union
from datetime import datetime
import os

# Import required components
from src.broker_module.upstox.data.CandleData import UpstoxHistoricalData
from .backward_testing import create_backtesting_engine, CandleData
from .strategy_evaluation_metrics import StrategyEvaluationMetrics
from .backtest_data_manager import BacktestDataManager

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class StrategyTester:
    def __init__(self, 
                 ticker: str,
                 timeframe: str,
                 testing_period_months: int,
                 strategy: Type,
                 strategy_params: Optional[Dict] = None,
                 initial_balance: float = 1000.0,
                 volume: float = 1.0,
                 leverage: float = 1.0,
                 commission: float = 0.0,
                 trade_engine_type: str = 'signals',
                 **kwargs):
        """
        Initialize strategy tester.
        
        Args:
            ticker: Trading symbol
            timeframe: Trading timeframe (e.g., '1min', '5min', '1hour', '1D')
            testing_period_months: Testing period in months
            strategy: Strategy class to test
            strategy_params: Parameters for the strategy
            initial_balance: Initial account balance
            volume: Trading volume
            leverage: Account leverage
            commission: Commission per trade
            trade_engine_type: Type of trade engine ('signals' or 'sltp')
            **kwargs: Additional parameters for backtesting
        """
        self.ticker = ticker
        self.timeframe = timeframe
        self.testing_period_months = testing_period_months
        self.strategy_class = strategy if isinstance(strategy, type) else type(strategy)
        self.strategy_params = strategy_params or {}
        self.initial_balance = initial_balance
        self.volume = volume
        self.leverage = leverage
        self.commission = commission
        self.trade_engine_type = trade_engine_type
        self.kwargs = kwargs
        
        # Initialize components
        self.data_loader = UpstoxHistoricalData()
        self.backtest_manager = BacktestDataManager()
        
        # Initialize results containers
        self.trades = pd.DataFrame()
        self.metrics = None
        self.stats = None
        self.final_recommendation = None

    def load_data(self) -> CandleData:
        """Load historical data using UpstoxHistoricalData."""
        try:
            # Calculate date range
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - pd.DateOffset(months=self.testing_period_months)).strftime('%Y-%m-%d')
            
            # Parse timeframe
            if 'min' in self.timeframe:
                unit = 'minutes'
                interval = int(self.timeframe.replace('min', ''))
            elif 'hour' in self.timeframe:
                unit = 'hours'
                interval = int(self.timeframe.replace('hour', ''))
            elif 'D' in self.timeframe:
                unit = 'days'
                interval = 1
            else:
                raise ValueError(f"Invalid timeframe: {self.timeframe}")
            
            # Load data
            candles = self.data_loader.get_historical_candles(
                instrument_key=self.ticker,
                unit=unit,
                interval=interval,
                to_date=end_date,
                from_date=start_date
            )
            
            # Convert to CandleData format
            return CandleData.from_list(candles)
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def run_backtest(self) -> Dict:
        """Run backtest and calculate metrics."""
        try:
            # Load data
            candle_data = self.load_data()
            
            # Create backtesting engine
            engine = create_backtesting_engine(
                candle_data=candle_data,
                engine_type=self.trade_engine_type,
                initial_balance=self.initial_balance,
                volume=self.volume,
                leverage=self.leverage,
                commission=self.commission,
                **self.kwargs
            )
            
            # Run backtest
            results = engine.run_backtest()
            
            # Calculate metrics
            metrics_calculator = StrategyEvaluationMetrics(
                backtest_results=results,
                ticker=self.ticker
            )
            
            # Store results
            self.trades = pd.DataFrame(results['trades'])
            self.metrics = metrics_calculator.get_metrics()
            self.stats = metrics_calculator.get_statistics()
            self.final_recommendation = metrics_calculator.get_recommendation()
            
            # Save results using backtest manager
            run_key = self.backtest_manager.save_backtest_results(
                results={
                    'trades': results['trades'],
                    'balance_history': results['balance_history'],
                    'equity_curve': results['equity_curve'],
                    'drawdown_curve': results['drawdown_curve'],
                    'risk_metrics': self.metrics,
                    'evaluation_metrics': self.stats,
                    'market_data': {
                        'timestamps': candle_data.timestamp.tolist(),
                        'prices': candle_data.close.tolist(),
                        'signals': candle_data.signal.tolist(),
                        'sl_levels': candle_data.sl.tolist() if candle_data.sl is not None else [],
                        'tp_levels': candle_data.tp.tolist() if candle_data.tp is not None else [],
                        'atr': [],  # Add ATR calculation if needed
                        'returns': np.diff(candle_data.close).tolist()
                    }
                },
                ticker=self.ticker,
                strategy_name=self.strategy_class.__name__
            )
            
            logger.info(f"Backtest completed successfully. Run key: {run_key}")
            
            return {
                'trades': self.trades,
                'metrics': self.metrics,
                'statistics': self.stats,
                'recommendation': self.final_recommendation,
                'run_key': run_key
            }
            
        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            raise

    def get_test_results(self, run_key: Optional[str] = None) -> Dict:
        """Get test results for a specific run or the latest run."""
        try:
            if run_key is None:
                run_key = self.backtest_manager.get_latest_run(self.strategy_class.__name__)
                if run_key is None:
                    raise ValueError("No backtest runs found")
            
            return self.backtest_manager.load_backtest_results(run_key)
            
        except Exception as e:
            logger.error(f"Error getting test results: {str(e)}")
            raise

    def get_strategy_runs(self) -> List[str]:
        """Get all run keys for the current strategy."""
        try:
            return self.backtest_manager.get_strategy_runs(self.strategy_class.__name__)
        except Exception as e:
            logger.error(f"Error getting strategy runs: {str(e)}")
            raise

def main():
    """
    Main function to test the StrategyTester class.
    This function demonstrates how to use the StrategyTester class with a sample strategy.
    """
    # Import a sample strategy for testing
    from src.strategy_module.strategies.moving_average_strategy import MovingAverageStrategy
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize the strategy tester
        tester = StrategyTester(
            ticker="RELIANCE",
            timeframe="1D",
            testing_period_months=6,
            strategy=MovingAverageStrategy,
            strategy_params={
                "ma_fast": 10,
                "ma_slow": 20,
                "rsi_period": 14,
                "rsi_overbought": 70,
                "rsi_oversold": 30
            },
            initial_balance=100000.0,  # Starting with 1 lakh
            volume=1.0,
            leverage=1.0,
            commission=0.05,  # 0.05% commission per trade
            trade_engine_type="signals"
        )
        
        # Run the backtest
        logger.info("Starting backtest...")
        results = tester.run_backtest()
        
        # Print results summary
        logger.info("\nBacktest Results Summary:")
        logger.info(f"Total Trades: {len(results['trades'])}")
        logger.info(f"Final Recommendation: {results['recommendation']}")
        
        # Print key metrics
        if results['metrics'] is not None:
            logger.info("\nKey Metrics:")
            for metric, value in results['metrics'].items():
                logger.info(f"{metric}: {value}")
        
        # Print key statistics
        if results['statistics'] is not None:
            logger.info("\nKey Statistics:")
            for stat, value in results['statistics'].items():
                logger.info(f"{stat}: {value}")
        
        # Get all runs for the strategy
        strategy_runs = tester.get_strategy_runs()
        logger.info(f"\nTotal strategy runs: {len(strategy_runs)}")
        
        # Get the latest test results
        latest_results = tester.get_test_results()
        logger.info("\nLatest test results retrieved successfully")
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    main()

