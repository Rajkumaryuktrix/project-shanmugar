import os
import sys
import logging
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime, timedelta
from colorama import Fore, Style
from tabulate import tabulate
import json
from pandas import Timestamp
import numpy as np
import argparse
from pathlib import Path

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from src.strategy_module.core_engine import StrategyEngine
from src.broker_module.upstox.utils.InstrumentKeyFinder import InstrumentKeyFinder
from src.broker_module.upstox.data.CandleData import UpstoxHistoricalData

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling datetime and Timestamp objects."""
    def default(self, obj):
        if isinstance(obj, (datetime, Timestamp)):
            return obj.isoformat()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

class StrategyCLI:
    """Command-line interface for strategy testing and optimization."""
    
    def __init__(self):
        """Initialize the CLI with the core engine."""
        self.engine = StrategyEngine()
        self.instrument_finder = InstrumentKeyFinder()
        self.json_encoder = DateTimeEncoder()
    
    def run_interactive(self):
        """Run the CLI in interactive mode."""
        try:
            while True:
                print(f"\n{Fore.CYAN}=== Strategy Testing and Optimization CLI ==={Style.RESET_ALL}")
                print("1. Run Backtest")
                print("2. Run Optimization")
                print("3. Exit")
                
                choice = input(f"\n{Fore.GREEN}Enter your choice (1-3): {Style.RESET_ALL}").strip()
                
                if choice == '1':
                    self._run_backtest_interactive()
                elif choice == '2':
                    self._run_optimization_interactive()
                elif choice == '3':
                    print(f"\n{Fore.GREEN}Exiting...{Style.RESET_ALL}")
                    break
                else:
                    print(f"{Fore.RED}Invalid choice! Please enter 1-3.{Style.RESET_ALL}")
        
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Program terminated by user.{Style.RESET_ALL}")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            print(f"\n{Fore.RED}An unexpected error occurred: {str(e)}{Style.RESET_ALL}")
    
    def _run_backtest_interactive(self):
        """Run backtest in interactive mode."""
        try:
            # 1. Select strategy
            strategy_name = None
            while True:
                try:
                    strategy_name = self._select_strategy()
                    if not strategy_name:
                        return
                    break
                except ValueError as e:
                    print(f"\n{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
                    continue
                except KeyboardInterrupt:
                    print("\n\nOperation cancelled by user.")
                    return
                except Exception as e:
                    print(f"\n{Fore.RED}Unexpected error: {str(e)}{Style.RESET_ALL}")
                    return

            # 2. Get instrument key
            instrument_key = self._get_instrument_key()
            if not instrument_key:
                return

            # 3. Configure data parameters
            data_params = self._configure_data_parameters()
            if not data_params:
                return

            # 4. Load data
            data = self._load_data(instrument_key, data_params)
            if data is None:
                return

            # 5. Configure strategy parameters
            strategy_params = self._configure_strategy_parameters(strategy_name)
            if not strategy_params:
                return

            # 6. Run strategy
            results = self.engine.run_strategy(
                data=data,
                strategy_name=strategy_name,
                strategy_params=strategy_params,
                tester_params={
                    'ticker': instrument_key,
                    'timeframe': data_params['interval'],
                    'testing_period_months': 1
                }
            )

            # 7. Display results
            self._display_results(results)

        except Exception as e:
            print(f"\n{Fore.RED}Error running backtest: {str(e)}{Style.RESET_ALL}")
            return
    
    def _run_optimization_interactive(self):
        """Run optimization in interactive mode."""
        try:
            # Similar structure to _run_backtest_interactive but for optimization
            # Implementation details here...
            pass
        except Exception as e:
            logger.error(f"Error in optimization: {str(e)}")
            print(f"\n{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
    
    def _get_instrument_key(self) -> Optional[str]:
        """Get instrument key from user input."""
        try:
            while True:
                symbol = input(f"\n{Fore.GREEN}Enter trading symbol (e.g., RELIANCE): {Style.RESET_ALL}").strip().upper()
                if not symbol:
                    print(f"{Fore.YELLOW}Symbol cannot be empty!{Style.RESET_ALL}")
                    continue
                    
                instrument_key = self.instrument_finder.find_instrument_key(symbol)
                if instrument_key:
                    print(f"{Fore.GREEN}Selected instrument key: {instrument_key}{Style.RESET_ALL}")
                    return instrument_key
                else:
                    print(f"{Fore.RED}Could not find instrument key for {symbol}{Style.RESET_ALL}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error finding instrument key: {str(e)}")
            print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
            return None
    
    def _configure_data_parameters(self) -> Optional[Dict]:
        """Configure data parameters interactively."""
        try:
            params = {}
            # Data type
            print(f"\n{Fore.CYAN}Select data type:{Style.RESET_ALL}")
            print("1. Historical Data")
            print("2. Intraday Data")
            while True:
                try:
                    choice = int(input(f"\n{Fore.GREEN}Enter choice (1-2): {Style.RESET_ALL}").strip())
                    if choice in [1, 2]:
                        params['data_type'] = 'historical' if choice == 1 else 'intraday'
                        break
                    print(f"{Fore.RED}Invalid choice!{Style.RESET_ALL}")
                except ValueError:
                    print(f"{Fore.RED}Please enter a valid number.{Style.RESET_ALL}")
            # Time unit
            print(f"\n{Fore.CYAN}Select time unit:{Style.RESET_ALL}")
            print("1. Minutes")
            print("2. Hours")
            print("3. Days")
            while True:
                try:
                    choice = int(input(f"\n{Fore.GREEN}Enter choice (1-3): {Style.RESET_ALL}").strip())
                    if choice in [1, 2, 3]:
                        params['unit'] = ['minutes', 'hours', 'days'][choice - 1]
                        break
                    print(f"{Fore.RED}Invalid choice!{Style.RESET_ALL}")
                except ValueError:
                    print(f"{Fore.RED}Please enter a valid number.{Style.RESET_ALL}")
            # Interval
            while True:
                try:
                    interval = int(input(f"\n{Fore.GREEN}Enter interval (e.g., 5 for 5 minutes): {Style.RESET_ALL}").strip())
                    if interval > 0:
                        params['interval'] = interval
                        break
                    print(f"{Fore.RED}Interval must be greater than 0!{Style.RESET_ALL}")
                except ValueError:
                    print(f"{Fore.RED}Please enter a valid number.{Style.RESET_ALL}")
            # Days (for historical data)
            if params['data_type'] == 'historical':
                while True:
                    try:
                        days = int(input(f"\n{Fore.GREEN}Enter number of days: {Style.RESET_ALL}").strip())
                        if days > 0:
                            params['days'] = days
                            break
                        print(f"{Fore.RED}Days must be greater than 0!{Style.RESET_ALL}")
                    except ValueError:
                        print(f"{Fore.RED}Please enter a valid number.{Style.RESET_ALL}")
            return params
        except Exception as e:
            logger.error(f"Error configuring data parameters: {str(e)}")
            print(f"\n{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
            return None
    
    def _configure_strategy_parameters(self, strategy_name: str) -> Optional[Dict]:
        """Configure strategy parameters interactively."""
        try:
            strategy_info = self.engine.get_available_strategies()[strategy_name]
            params = {}
            print(f"\n{Fore.CYAN}=== {strategy_name.replace('_', ' ').title()} Parameters ==={Style.RESET_ALL}")
            # First, handle strategy-specific parameters
            for param_name, param_info in strategy_info['params'].items():
                while True:
                    print(f"\n{Fore.GREEN}{param_name} ({param_info['type']}):{Style.RESET_ALL}")
                    print(f"Description: {param_info['description']}")
                    if param_info['default'] is not None:
                        print(f"Default value: {param_info['default']}")
                    value = input(f"{Fore.CYAN}Enter value (or press Enter for default): {Style.RESET_ALL}").strip()
                    if not value and param_info['default'] is not None:
                        value = param_info['default']
                        print(f"Using default value: {value}")
                    try:
                        # Convert value based on parameter type
                        if param_info['type'] == 'int':
                            value = int(value)
                        elif param_info['type'] == 'float':
                            value = float(value)
                        elif param_info['type'] == 'bool':
                            value = value.lower() in ['true', 'yes', '1']
                        elif param_info['type'] == 'list':
                            value = [x.strip() for x in value.split(',')]
                        params[param_name] = value
                        break
                    except ValueError:
                        print(f"{Fore.RED}Invalid value. Please enter a valid {param_info['type']}.{Style.RESET_ALL}")
            # Now handle core engine parameters
            print(f"\n{Fore.CYAN}=== Core Engine Parameters ==={Style.RESET_ALL}")
            # Backtesting parameters
            print(f"\n{Fore.GREEN}Backtesting Parameters:{Style.RESET_ALL}")
            params['forward_test'] = self._get_bool_input("Run forward testing? (y/n): ", default=True)
            params['trade_engine_type'] = self._get_choice_input("Select trade engine type:", ["Signals", "SLTP"], default="Signals")
            params['investing_amount'] = self._get_float_input("Initial investment amount: ", default=1000.0)
            params['account_leverage'] = self._get_float_input("Account leverage: ", default=500.0)
            params['volume'] = self._get_int_input("Trading volume: ", default=1)
            params['commission'] = self._get_float_input("Commission per trade: ", default=7.0)
            # Risk management parameters
            print(f"\n{Fore.GREEN}Risk Management Parameters:{Style.RESET_ALL}")
            params['max_position_size'] = self._get_float_input("Maximum position size (0-1): ", default=0.1)
            params['max_drawdown'] = self._get_float_input("Maximum drawdown (0-1): ", default=0.2)
            params['risk_per_trade'] = self._get_float_input("Risk per trade (0-1): ", default=0.02)
            # Trade engine parameters
            print(f"\n{Fore.GREEN}Trade Engine Parameters:{Style.RESET_ALL}")
            params['stop_loss_pct'] = self._get_float_input("Stop loss percentage: ", default=2.0)
            params['take_profit_pct'] = self._get_float_input("Take profit percentage: ", default=4.0)
            params['trailing_stop'] = self._get_bool_input("Use trailing stop? (y/n): ", default=False)
            if params['trailing_stop']:
                params['trailing_stop_pct'] = self._get_float_input("Trailing stop percentage: ", default=1.0)
            # Backward testing parameters
            print(f"\n{Fore.GREEN}Backward Testing Parameters:{Style.RESET_ALL}")
            params['separate_close_signals'] = self._get_bool_input("Handle close signals separately? (y/n): ", default=False)
            params['parallel_opening'] = self._get_bool_input("Allow parallel position opening? (y/n): ", default=False)
            params['max_open_positions'] = self._get_int_input("Maximum open positions: ", default=1)
            params['position_sizing_method'] = self._get_choice_input("Position sizing method:", ["fixed", "risk_based", "kelly"], default="fixed")
            params['allow_shorting'] = self._get_bool_input("Allow shorting? (y/n): ", default=True)
            params['allow_partial_fills'] = self._get_bool_input("Allow partial fills? (y/n): ", default=False)
            # Slippage parameters
            params['slippage_model'] = self._get_choice_input("Slippage model:", ["none", "fixed", "percentage", "random"], default="none")
            if params['slippage_model'] != "none":
                params['slippage_value'] = self._get_float_input("Slippage value: ", default=0.0)
            # Statistical testing parameters
            print(f"\n{Fore.GREEN}Statistical Testing Parameters:{Style.RESET_ALL}")
            params['significance_level'] = self._get_float_input("Significance level (0-1): ", default=0.05)
            params['monte_carlo_simulations'] = self._get_int_input("Number of Monte Carlo simulations: ", default=1000)
            params['confidence_level'] = self._get_float_input("Confidence level (0-1): ", default=0.95)
            # Performance metrics parameters
            print(f"\n{Fore.GREEN}Performance Metrics Parameters:{Style.RESET_ALL}")
            params['benchmark_symbol'] = input(f"{Fore.CYAN}Enter benchmark symbol (optional): {Style.RESET_ALL}").strip() or None
            params['risk_free_rate'] = self._get_float_input("Risk-free rate: ", default=0.05)
            return params
        except Exception as e:
            logger.error(f"Error configuring strategy parameters: {str(e)}")
            print(f"\n{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
            return None
    
    def _get_bool_input(self, prompt: str, default: bool = True) -> bool:
        """Get boolean input from user."""
        while True:
            value = input(f"{Fore.CYAN}{prompt}{Style.RESET_ALL}").strip().lower()
            if not value:
                return default
            if value in ['y', 'yes', 'true', '1']:
                return True
            if value in ['n', 'no', 'false', '0']:
                return False
            print(f"{Fore.RED}Invalid input. Please enter y/n.{Style.RESET_ALL}")
    
    def _get_choice_input(self, prompt: str, choices: List[str], default: str) -> str:
        """Get choice input from user."""
        print(f"\n{Fore.CYAN}{prompt}{Style.RESET_ALL}")
        for i, choice in enumerate(choices, 1):
            print(f"{i}. {choice}")
        
        while True:
            try:
                value = input(f"{Fore.CYAN}Enter choice (1-{len(choices)}): {Style.RESET_ALL}").strip()
                if not value:
                    return default
                choice = int(value)
                if 1 <= choice <= len(choices):
                    return choices[choice - 1]
                print(f"{Fore.RED}Invalid choice. Please enter 1-{len(choices)}.{Style.RESET_ALL}")
            except ValueError:
                print(f"{Fore.RED}Please enter a valid number.{Style.RESET_ALL}")
        
    def _get_float_input(self, prompt: str, default: float) -> float:
        """Get float input from user."""
        while True:
            try:
                value = input(f"{Fore.CYAN}{prompt}{Style.RESET_ALL}").strip()
                if not value:
                    return default
                return float(value)
            except ValueError:
                print(f"{Fore.RED}Please enter a valid number.{Style.RESET_ALL}")
        
    def _get_int_input(self, prompt: str, default: int) -> int:
        """Get integer input from user."""
        while True:
            try:
                value = input(f"{Fore.CYAN}{prompt}{Style.RESET_ALL}").strip()
                if not value:
                    return default
                return int(value)
            except ValueError:
                print(f"{Fore.RED}Please enter a valid integer.{Style.RESET_ALL}")
    
    def _display_results(self, results: Dict):
        """Display strategy results."""
        try:
            print(f"\n{Fore.GREEN}Strategy Test Results:{Style.RESET_ALL}")
            print("-" * 50)
            
            if 'trades' in results:
                print(f"\nTotal Trades: {len(results['trades'])}")
            
            if 'metrics' in results:
                print("\nPerformance Metrics:")
                metrics_df = pd.DataFrame(results['metrics'])
                print(tabulate(metrics_df, headers='keys', tablefmt='psql'))
            
            if 'statistics' in results:
                print("\nStatistics:")
                stats_df = pd.DataFrame(results['statistics'])
                print(tabulate(stats_df, headers='keys', tablefmt='psql'))
            
            if 'recommendation' in results:
                print(f"\nRecommendation: {results['recommendation']}")
            
        except Exception as e:
            logger.error(f"Error displaying results: {str(e)}")
            print(f"\n{Fore.RED}Error displaying results: {str(e)}{Style.RESET_ALL}")

    def _select_strategy(self) -> Optional[str]:
        """Select a strategy from available options."""
        try:
            # Get available strategies
            available_strategies = self.engine.get_available_strategies()
            if not available_strategies:
                print(f"{Fore.RED}No strategies available!{Style.RESET_ALL}")
                return None

            # Display available strategies
            print(f"\n{Fore.CYAN}Available Strategies:{Style.RESET_ALL}")
            for i, (strategy_name, strategy_info) in enumerate(available_strategies.items(), 1):
                print(f"{i}. {strategy_name.replace('_', ' ').title()}")
                print(f"   Description: {strategy_info.get('description', 'No description available')}")

            # Get user choice
            while True:
                try:
                    choice = input(f"\n{Fore.GREEN}Select strategy (1-{len(available_strategies)}): {Style.RESET_ALL}").strip()
                    if not choice:
                        print(f"{Fore.YELLOW}Strategy selection cancelled.{Style.RESET_ALL}")
                        return None

                    choice = int(choice)
                    if 1 <= choice <= len(available_strategies):
                        strategy_name = list(available_strategies.keys())[choice - 1]
                        print(f"{Fore.GREEN}Selected strategy: {strategy_name.replace('_', ' ').title()}{Style.RESET_ALL}")
                        return strategy_name
                    else:
                        print(f"{Fore.RED}Invalid choice! Please enter a number between 1 and {len(available_strategies)}.{Style.RESET_ALL}")
                except ValueError:
                    print(f"{Fore.RED}Please enter a valid number.{Style.RESET_ALL}")

        except Exception as e:
            logger.error(f"Error selecting strategy: {str(e)}")
            raise

    def _load_data(self, instrument_key: str, data_params: Dict) -> Optional[pd.DataFrame]:
        """Load historical data for the selected instrument."""
        try:
            print(f"\n{Fore.CYAN}Loading historical data...{Style.RESET_ALL}")
            
            # Get historical data
            historical_data = UpstoxHistoricalData()
            data = historical_data.get_historical_candles(
                instrument_key=instrument_key,
                unit=data_params['unit'],
                interval=data_params['interval'],
                to_date=datetime.now().strftime('%Y-%m-%d'),
                from_date=(datetime.now() - timedelta(days=data_params.get('days', 30))).strftime('%Y-%m-%d')
            )
            
            if data.empty:
                print(f"{Fore.RED}No data retrieved for the specified parameters.{Style.RESET_ALL}")
                return None
            
            # Ensure all required columns are present
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in data.columns:
                    print(f"{Fore.RED}Missing required column: {col}{Style.RESET_ALL}")
                    return None
            
            # Convert time to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
                data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Convert numeric columns to float
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                data[col] = data[col].astype(float)
            
            # Sort by time
            data = data.sort_values('timestamp')
            
            # Rename columns to match expected format
            data = data.rename(columns={
                'timestamp': 'time',
                'close': 'Price'  # Rename close to Price as expected by strategy tester
            })
            
            print(f"{Fore.GREEN}Successfully loaded {len(data)} rows of historical data.{Style.RESET_ALL}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            print(f"\n{Fore.RED}Error loading data: {str(e)}{Style.RESET_ALL}")
            return None

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Trading Strategy CLI')
    
    # Strategy selection
    parser.add_argument('--strategy', type=str, required=True,
                      choices=['rsi', 'moving_average', 'time_based'],
                      help='Strategy to use')
    
    # Data parameters
    parser.add_argument('--data-file', type=str, required=True,
                      help='Path to data file (CSV)')
    
    # Strategy-specific parameters
    parser.add_argument('--rsi-period', type=int, default=14,
                      help='RSI period (for RSI strategy)')
    parser.add_argument('--overbought', type=float, default=70,
                      help='Overbought level (for RSI strategy)')
    parser.add_argument('--oversold', type=float, default=30,
                      help='Oversold level (for RSI strategy)')
    
    parser.add_argument('--fast-period', type=int, default=20,
                      help='Fast MA period (for Moving Average strategy)')
    parser.add_argument('--slow-period', type=int, default=50,
                      help='Slow MA period (for Moving Average strategy)')
    
    parser.add_argument('--entry-times', type=str, nargs='+',
                      help='Entry times in HH:MM format (for Time-based strategy)')
    parser.add_argument('--exit-times', type=str, nargs='+',
                      help='Exit times in HH:MM format (for Time-based strategy)')
    
    # Common parameters
    parser.add_argument('--sl-atr-multiplier', type=float, default=2.0,
                      help='Stop loss ATR multiplier')
    parser.add_argument('--tp-atr-multiplier', type=float, default=3.0,
                      help='Take profit ATR multiplier')
    parser.add_argument('--atr-period', type=int, default=14,
                      help='ATR period')
    
    # Output parameters
    parser.add_argument('--output-file', type=str,
                      help='Path to save results (CSV)')
    
    return parser.parse_args()

def get_strategy_params(args: argparse.Namespace) -> Dict[str, Any]:
    """Get strategy-specific parameters from arguments."""
    params = {
        'sl_atr_multiplier': args.sl_atr_multiplier,
        'tp_atr_multiplier': args.tp_atr_multiplier,
        'atr_period': args.atr_period
    }
    
    if args.strategy == 'rsi':
        params.update({
            'rsi_period': args.rsi_period,
            'overbought': args.overbought,
            'oversold': args.oversold
        })
    elif args.strategy == 'moving_average':
        params.update({
            'fast_period': args.fast_period,
            'slow_period': args.slow_period
        })
    elif args.strategy == 'time_based':
        if not args.entry_times or not args.exit_times:
            raise ValueError("Entry and exit times are required for time-based strategy")
        params.update({
            'entry_times': args.entry_times,
            'exit_times': args.exit_times
        })
    
    return params

def main():
    """Main CLI entry point."""
    try:
        # Check if any arguments were provided
        if len(sys.argv) > 1:
            # Command-line mode
            args = parse_args()
            
            # Initialize engine
            engine = StrategyEngine()
            
            # Load strategy
            strategy_params = get_strategy_params(args)
            engine.load_strategy(args.strategy, **strategy_params)
            
            # Load data
            data = pd.read_csv(args.data_file)
            engine.load_data(data)
            
            # Run strategy
            results = engine.run_strategy()
            
            # Save results
            if args.output_file:
                engine.save_results(results, args.output_file)
            else:
                engine.save_results(results)
            
            logger.info("Strategy execution completed successfully")
        else:
            # Interactive mode
            cli = StrategyCLI()
            cli.run_interactive()
        
    except Exception as e:
        logger.error(f"Error in strategy execution: {str(e)}")
        raise

if __name__ == '__main__':
    main() 