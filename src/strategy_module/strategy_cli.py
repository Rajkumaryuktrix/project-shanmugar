import os
import sys
import logging
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime, timedelta
from colorama import Fore, Style
from tabulate import tabulate
import json
from pandas import Timestamp
import numpy as np

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from src.strategy_module.core_engine import StrategyEngine
from src.broker_module.upstox.utils.InstrumentKeyFinder import InstrumentKeyFinder

# Configure logging
logging.basicConfig(level=logging.INFO)
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
    """
    Command-line interface for strategy testing and optimization.
    Provides a user-friendly interface to the core engine.
    """
    
    def __init__(self):
        """Initialize the CLI with the core engine."""
        self.engine = StrategyEngine()
        self.instrument_finder = InstrumentKeyFinder()
        self.history = []  # For navigation history
        self.json_encoder = DateTimeEncoder()
    
    def _display_menu(self, title: str, options: List[tuple], allow_back: bool = True, allow_skip: bool = False) -> Optional[str]:
        """Display a menu and get user choice."""
        print(f"\n{Fore.CYAN}=== {title} ==={Style.RESET_ALL}")
        
        for i, (key, label) in enumerate(options, 1):
            print(f"{i}. {label}")
        
        if allow_back:
            print("0. Back")
        if allow_skip:
            print("Enter. Skip")
        
        while True:
            choice = input(f"\n{Fore.GREEN}Enter your choice: {Style.RESET_ALL}").strip()
            if allow_skip and not choice:  # Allow skipping if enabled
                return None
            try:
                choice = int(choice)
                if choice == 0 and allow_back:
                    return 'back'
                if 1 <= choice <= len(options):
                    return options[choice - 1][0]
                print(f"{Fore.RED}Invalid choice!{Style.RESET_ALL}")
            except ValueError:
                print(f"{Fore.RED}Please enter a valid number.{Style.RESET_ALL}")
    
    def _get_instrument_key(self) -> Optional[str]:
        """
        Get instrument key from user input using InstrumentKeyFinder.
        
        Returns:
            Optional[str]: Selected instrument key or None if user cancels
        """
        while True:
            try:
                symbol = input(f"\n{Fore.GREEN}Enter trading symbol (e.g., RELIANCE): {Style.RESET_ALL}").strip().upper()
                if not symbol:
                    print(f"{Fore.YELLOW}Symbol cannot be empty!{Style.RESET_ALL}")
                    continue
                    
                # Use find_instrument_key which handles both exact and similar matches
                instrument_key = self.instrument_finder.find_instrument_key(symbol)
                if instrument_key:
                    print(f"{Fore.GREEN}Selected instrument key: {instrument_key}{Style.RESET_ALL}")
                    return instrument_key
                    
            except ValueError as e:
                if str(e) == "Search cancelled by user":
                    print(f"{Fore.YELLOW}Search cancelled by user{Style.RESET_ALL}")
                    return None
                logger.error(f"Error validating symbol: {str(e)}")
                print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                print(f"{Fore.RED}Unexpected error: {str(e)}{Style.RESET_ALL}")
    
    def _configure_data_parameters(self) -> Optional[Dict]:
        """Configure data parameters with interactive menu."""
        print(f"\n{Fore.CYAN}=== Data Parameters ==={Style.RESET_ALL}")
        
        # Data type
        data_type_options = [
            ('historical', 'Historical Data'),
            ('intraday', 'Intraday Data (Current Day)')
        ]
        data_type = self._display_menu("Select Data Type", data_type_options)
        if data_type == 'back':
            return None
        
        # Time unit
        unit_options = [
            ('minutes', 'Minutes'),
            ('hours', 'Hours'),
            ('days', 'Days')
        ]
        unit = self._display_menu("Select Time Unit", unit_options)
        if unit == 'back':
            return None
        
        # Interval
        while True:
            try:
                interval = int(input(f"\n{Fore.GREEN}Enter time interval: {Style.RESET_ALL}").strip())
                if interval > 0:
                    break
                print(f"{Fore.RED}Interval must be greater than 0.{Style.RESET_ALL}")
            except ValueError:
                print(f"{Fore.RED}Please enter a valid number.{Style.RESET_ALL}")
        
        params = {
            'unit': unit,
            'interval': interval,
            'data_type': data_type
        }
        
        # Only ask for days if historical data is selected
        if data_type == 'historical':
            while True:
                try:
                    days = int(input(f"\n{Fore.GREEN}Enter number of days for historical data: {Style.RESET_ALL}").strip())
                    if days > 0:
                        params['days'] = days
                        break
                    print(f"{Fore.RED}Number of days must be greater than 0.{Style.RESET_ALL}")
                except ValueError:
                    print(f"{Fore.RED}Please enter a valid number.{Style.RESET_ALL}")
        
        return params
    
    def _configure_strategy_parameters(self, strategy_name: str) -> Optional[Dict]:
        """Configure strategy parameters with interactive menu."""
        strategy_info = self.engine.get_available_strategies()[strategy_name]
        params = {}
        
        print(f"\n{Fore.CYAN}=== {strategy_name.replace('_', ' ').title()} Parameters ==={Style.RESET_ALL}")
        
        for param_name, param_info in strategy_info['params'].items():
            while True:
                print(f"\n{Fore.GREEN}{param_name} ({param_info['type']}):{Style.RESET_ALL}")
                print(f"Description: {param_info['description']}")
                if param_info['default'] is not None:
                    print(f"Default value: {param_info['default']}")
                
                value = input(f"{Fore.CYAN}Enter value (or 'back' to return): {Style.RESET_ALL}").strip()
                
                if value.lower() == 'back':
                    return None
                
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
                    elif param_info['type'] == 'str':
                        value = str(value)
                    
                    # Validate numeric ranges if specified
                    if isinstance(value, (int, float)):
                        if 'min' in param_info and value < param_info['min']:
                            print(f"{Fore.RED}Value must be greater than or equal to {param_info['min']}.{Style.RESET_ALL}")
                            continue
                        if 'max' in param_info and value > param_info['max']:
                            print(f"{Fore.RED}Value must be less than or equal to {param_info['max']}.{Style.RESET_ALL}")
                            continue
                    
                    params[param_name] = value
                    break
                except ValueError:
                    print(f"{Fore.RED}Invalid value. Please enter a valid {param_info['type']}.{Style.RESET_ALL}")
        
        return params
    
    def _configure_tester_parameters(self) -> Optional[Dict]:
        """Configure tester parameters with interactive menu."""
        print(f"\n{Fore.CYAN}=== Tester Parameters ==={Style.RESET_ALL}")
        
        params = {}
        
        # Required Parameters
        print(f"\n{Fore.YELLOW}=== Required Parameters ==={Style.RESET_ALL}")
        
        # Testing period (Required)
        while True:
            try:
                testing_period = input(f"\n{Fore.GREEN}Enter testing period in months (e.g., 3) [Required]: {Style.RESET_ALL}").strip()
                if not testing_period:
                    print(f"{Fore.RED}Testing period is required!{Style.RESET_ALL}")
                    continue
                testing_period = int(testing_period)
                if testing_period > 0:
                    params['testing_period_months'] = testing_period
                    break
                print(f"{Fore.RED}Period must be greater than 0.{Style.RESET_ALL}")
            except ValueError:
                print(f"{Fore.RED}Please enter a valid number.{Style.RESET_ALL}")
        
        # Optional Parameters
        print(f"\n{Fore.YELLOW}=== Optional Parameters ==={Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Press Enter to skip optional parameters{Style.RESET_ALL}")
        
        optional_params = {
            'investment': {
                'prompt': 'Enter investment amount (e.g., 1000)',
                'type': 'float',
                'validation': lambda x: x > 0,
                'error_msg': 'Amount must be greater than 0',
                'param_name': 'investing_amount'
            },
            'leverage': {
                'prompt': 'Enter account leverage (e.g., 500)',
                'type': 'float',
                'validation': lambda x: x > 0,
                'error_msg': 'Leverage must be greater than 0',
                'param_name': 'account_leverage'
            },
            'volume': {
                'prompt': 'Enter trading volume (e.g., 0.01)',
                'type': 'int',
                'validation': lambda x: 1 < x <= 1000,
                'error_msg': 'Volume must be between 1 and 1000',
                'param_name': 'volume'
            },
            'commission': {
                'prompt': 'Enter commission percentage (e.g., 7.0)',
                'type': 'float',
                'validation': lambda x: x >= 0,
                'error_msg': 'Commission must be non-negative',
                'param_name': 'commission'
            },
            'stop_loss': {
                'prompt': 'Enter stop loss percentage (e.g., 2.0)',
                'type': 'float',
                'validation': lambda x: x > 0,
                'error_msg': 'Stop loss must be greater than 0',
                'param_name': 'stop_loss_pct'
            },
            'take_profit': {
                'prompt': 'Enter take profit percentage (e.g., 4.0)',
                'type': 'float',
                'validation': lambda x: x > 0,
                'error_msg': 'Take profit must be greater than 0',
                'param_name': 'take_profit_pct'
            }
        }
        
        # Configure optional parameters
        while True:
            print(f"\n{Fore.CYAN}Select optional parameter to configure:{Style.RESET_ALL}")
            options = [(key, f"{key.replace('_', ' ').title()}") for key in optional_params.keys()]
            options.append(('done', 'Done configuring optional parameters'))
            
            choice = self._display_menu("Optional Parameters", options, allow_back=False)
            
            if choice == 'done':
                break
                
            param_config = optional_params[choice]
            
            while True:
                try:
                    value = input(f"\n{Fore.GREEN}{param_config['prompt']} [Optional]: {Style.RESET_ALL}").strip()
                    if not value:  # Skip if empty
                        break
                        
                    if param_config['type'] == 'float':
                        value = float(value)
                    elif param_config['type'] == 'int':
                        value = int(value)
                        
                    if param_config['validation'](value):
                        params[param_config['param_name']] = value
                        break
                    print(f"{Fore.RED}{param_config['error_msg']}{Style.RESET_ALL}")
                except ValueError:
                    print(f"{Fore.RED}Please enter a valid {param_config['type']}.{Style.RESET_ALL}")
        
        return params
    
    def _configure_optimizer_parameters(self) -> Optional[Dict]:
        """Configure optimizer parameters with interactive menu."""
        print(f"\n{Fore.CYAN}=== Optimizer Parameters ==={Style.RESET_ALL}")
        
        # Optimization method
        method_options = [
            ('grid_search', 'Grid Search'),
            ('differential_evolution', 'Differential Evolution')
        ]
        method = self._display_menu("Select Optimization Method", method_options)
        if method == 'back':
            return None
        
        # Optimization criterion
        criterion_options = [
            ('Balance_Max', 'Maximum Balance'),
            ('Win_Rate_Max', 'Maximum Win Rate'),
            ('Profit_Factor_Max', 'Maximum Profit Factor')
        ]
        criterion = self._display_menu("Select Optimization Criterion", criterion_options)
        if criterion == 'back':
            return None
        
        # Max combinations
        while True:
            try:
                max_combinations = int(input(f"\n{Fore.GREEN}Enter maximum number of combinations to test (e.g., 1000): {Style.RESET_ALL}").strip())
                if max_combinations > 0:
                    break
                print(f"{Fore.RED}Number must be greater than 0.{Style.RESET_ALL}")
            except ValueError:
                print(f"{Fore.RED}Please enter a valid number.{Style.RESET_ALL}")
        
        return {
            'method': method,
            'criterion': criterion,
            'max_combinations': max_combinations
        }
    
    def _configure_backtesting_parameters(self) -> Optional[Dict]:
        """Configure backtesting parameters with interactive menu."""
        print(f"\n{Fore.CYAN}=== Backtesting Parameters ==={Style.RESET_ALL}")
        
        params = {}
        
        # Optional Parameters
        print(f"\n{Fore.YELLOW}=== Optional Parameters ==={Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Press Enter to skip optional parameters{Style.RESET_ALL}")
        
        optional_params = {
            'forward_test': {
                'prompt': 'Run forward testing? (y/n)',
                'type': 'bool',
                'param_name': 'forward_test'
            },
            'engine_type': {
                'prompt': 'Select trade engine type',
                'type': 'menu',
                'options': [
                    ('Signals', 'Signal-based Trading'),
                    ('SLTP', 'Stop Loss/Take Profit Trading')
                ],
                'param_name': 'trade_engine_type'
            },
            'base_price': {
                'prompt': 'Select base price',
                'type': 'menu',
                'options': [
                    ('open', 'Open Price'),
                    ('high', 'High Price'),
                    ('low', 'Low Price'),
                    ('close', 'Close Price')
                ],
                'param_name': 'base_price'
            }
        }
        
        # Configure optional parameters
        while True:
            print(f"\n{Fore.CYAN}Select optional parameter to configure:{Style.RESET_ALL}")
            options = [(key, f"{key.replace('_', ' ').title()}") for key in optional_params.keys()]
            options.append(('done', 'Done configuring optional parameters'))
            
            choice = self._display_menu("Optional Parameters", options, allow_back=False)
            
            if choice == 'done':
                break
                
            param_config = optional_params[choice]
            
            if param_config['type'] == 'bool':
                value = self._get_boolean_input(f"{param_config['prompt']} [Optional]: ", allow_skip=True)
                if value is not None:
                    params[param_config['param_name']] = value
                    
            elif param_config['type'] == 'menu':
                value = self._display_menu(param_config['prompt'], param_config['options'], allow_skip=True)
                if value and value != 'back':
                    params[param_config['param_name']] = value
        
        return params
    
    def _configure_backward_testing_parameters(self) -> Optional[Dict]:
        """Configure backward testing specific parameters."""
        print(f"\n{Fore.CYAN}=== Backward Testing Parameters ==={Style.RESET_ALL}")
        
        params = {}
        
        # Optional Parameters
        print(f"\n{Fore.YELLOW}=== Optional Parameters ==={Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Press Enter to skip optional parameters{Style.RESET_ALL}")
        
        optional_params = {
            'separate_close': {
                'prompt': 'Handle close signals separately? (y/n)',
                'type': 'bool',
                'param_name': 'separate_close_signals'
            },
            'parallel': {
                'prompt': 'Allow parallel position opening? (y/n)',
                'type': 'bool',
                'param_name': 'parallel_opening'
            },
            'max_positions': {
                'prompt': 'Enter maximum number of open positions',
                'type': 'int',
                'validation': lambda x: x > 0,
                'error_msg': 'Number must be greater than 0',
                'param_name': 'max_open_positions'
            },
            'sizing_method': {
                'prompt': 'Select position sizing method',
                'type': 'menu',
                'options': [
                    ('fixed', 'Fixed Size'),
                    ('risk_based', 'Risk-based'),
                    ('kelly', 'Kelly Criterion')
                ],
                'param_name': 'position_sizing_method'
            },
            'allow_short': {
                'prompt': 'Allow short positions? (y/n)',
                'type': 'bool',
                'param_name': 'allow_shorting'
            },
            'allow_partial': {
                'prompt': 'Allow partial order fills? (y/n)',
                'type': 'bool',
                'param_name': 'allow_partial_fills'
            },
            'slippage': {
                'prompt': 'Select slippage model',
                'type': 'menu',
                'options': [
                    ('none', 'No Slippage'),
                    ('fixed', 'Fixed Slippage'),
                    ('percentage', 'Percentage Slippage'),
                    ('random', 'Random Slippage')
                ],
                'param_name': 'slippage_model',
                'dependent_param': {
                    'name': 'slippage_value',
                    'prompt': 'Enter slippage value',
                    'type': 'float',
                    'validation': lambda x: x >= 0,
                    'error_msg': 'Value must be non-negative'
                }
            }
        }
        
        # Configure optional parameters
        while True:
            print(f"\n{Fore.CYAN}Select optional parameter to configure:{Style.RESET_ALL}")
            options = [(key, f"{key.replace('_', ' ').title()}") for key in optional_params.keys()]
            options.append(('done', 'Done configuring optional parameters'))
            
            choice = self._display_menu("Optional Parameters", options, allow_back=False)
            
            if choice == 'done':
                break
                
            param_config = optional_params[choice]
            
            if param_config['type'] == 'bool':
                value = self._get_boolean_input(f"{param_config['prompt']} [Optional]: ", allow_skip=True)
                if value is not None:
                    params[param_config['param_name']] = value
                    
            elif param_config['type'] == 'menu':
                value = self._display_menu(param_config['prompt'], param_config['options'], allow_skip=True)
                if value and value != 'back':
                    params[param_config['param_name']] = value
                    
                    # Handle dependent parameter if exists
                    if 'dependent_param' in param_config and value != 'none':
                        dep_param = param_config['dependent_param']
                        while True:
                            try:
                                dep_value = input(f"\n{Fore.GREEN}{dep_param['prompt']} [Optional]: {Style.RESET_ALL}").strip()
                                if not dep_value:  # Skip if empty
                                    break
                                    
                                if dep_param['type'] == 'float':
                                    dep_value = float(dep_value)
                                elif dep_param['type'] == 'int':
                                    dep_value = int(dep_value)
                                    
                                if dep_param['validation'](dep_value):
                                    params[dep_param['name']] = dep_value
                                    break
                                print(f"{Fore.RED}{dep_param['error_msg']}{Style.RESET_ALL}")
                            except ValueError:
                                print(f"{Fore.RED}Please enter a valid {dep_param['type']}.{Style.RESET_ALL}")
                                
            else:  # numeric types
                while True:
                    try:
                        value = input(f"\n{Fore.GREEN}{param_config['prompt']} [Optional]: {Style.RESET_ALL}").strip()
                        if not value:  # Skip if empty
                            break
                            
                        if param_config['type'] == 'float':
                            value = float(value)
                        elif param_config['type'] == 'int':
                            value = int(value)
                            
                        if param_config['validation'](value):
                            params[param_config['param_name']] = value
                            break
                        print(f"{Fore.RED}{param_config['error_msg']}{Style.RESET_ALL}")
                    except ValueError:
                        print(f"{Fore.RED}Please enter a valid {param_config['type']}.{Style.RESET_ALL}")
        
        return params
    
    def _configure_statistical_parameters(self) -> Optional[Dict]:
        """Configure statistical testing parameters."""
        print(f"\n{Fore.CYAN}=== Statistical Testing Parameters ==={Style.RESET_ALL}")
        
        params = {}
        
        # Significance level
        while True:
            try:
                significance = float(input(f"\n{Fore.GREEN}Enter significance level (0-1): {Style.RESET_ALL}").strip())
                if 0 < significance < 1:
                    params['significance_level'] = significance
                    break
                print(f"{Fore.RED}Value must be between 0 and 1.{Style.RESET_ALL}")
            except ValueError:
                print(f"{Fore.RED}Please enter a valid number.{Style.RESET_ALL}")
        
        # Monte Carlo simulations
        while True:
            try:
                simulations = int(input(f"\n{Fore.GREEN}Enter number of Monte Carlo simulations: {Style.RESET_ALL}").strip())
                if simulations > 0:
                    params['monte_carlo_simulations'] = simulations
                    break
                print(f"{Fore.RED}Number must be greater than 0.{Style.RESET_ALL}")
            except ValueError:
                print(f"{Fore.RED}Please enter a valid number.{Style.RESET_ALL}")
        
        # Confidence level
        while True:
            try:
                confidence = float(input(f"\n{Fore.GREEN}Enter confidence level (0-1): {Style.RESET_ALL}").strip())
                if 0 < confidence < 1:
                    params['confidence_level'] = confidence
                    break
                print(f"{Fore.RED}Value must be between 0 and 1.{Style.RESET_ALL}")
            except ValueError:
                print(f"{Fore.RED}Please enter a valid number.{Style.RESET_ALL}")
        
        return params
    
    def _configure_trade_engine_parameters(self) -> Optional[Dict]:
        """Configure trade engine parameters."""
        print(f"\n{Fore.CYAN}=== Trade Engine Parameters ==={Style.RESET_ALL}")
        
        params = {}
        
        # Stop loss and take profit
        while True:
            try:
                stop_loss = float(input(f"\n{Fore.GREEN}Enter stop loss percentage: {Style.RESET_ALL}").strip())
                take_profit = float(input(f"{Fore.GREEN}Enter take profit percentage: {Style.RESET_ALL}").strip())
                if 0 < stop_loss < take_profit:
                    params['stop_loss_pct'] = stop_loss
                    params['take_profit_pct'] = take_profit
                    break
                print(f"{Fore.RED}Stop loss must be less than take profit.{Style.RESET_ALL}")
            except ValueError:
                print(f"{Fore.RED}Please enter valid numbers.{Style.RESET_ALL}")
        
        # Trailing stop
        params['trailing_stop'] = self._get_boolean_input("Use trailing stop? (y/n): ")
        
        if params['trailing_stop']:
            while True:
                try:
                    trailing_pct = float(input(f"\n{Fore.GREEN}Enter trailing stop percentage: {Style.RESET_ALL}").strip())
                    if trailing_pct > 0:
                        params['trailing_stop_pct'] = trailing_pct
                        break
                    print(f"{Fore.RED}Value must be greater than 0.{Style.RESET_ALL}")
                except ValueError:
                    print(f"{Fore.RED}Please enter a valid number.{Style.RESET_ALL}")
        
        return params
    
    def _configure_risk_management_parameters(self) -> Optional[Dict]:
        """Configure risk management parameters."""
        print(f"\n{Fore.CYAN}=== Risk Management Parameters ==={Style.RESET_ALL}")
        
        params = {}
        
        # Max position size
        while True:
            try:
                max_size = float(input(f"\n{Fore.GREEN}Enter maximum position size (0-1): {Style.RESET_ALL}").strip())
                if 0 < max_size <= 1:
                    params['max_position_size'] = max_size
                    break
                print(f"{Fore.RED}Value must be between 0 and 1.{Style.RESET_ALL}")
            except ValueError:
                print(f"{Fore.RED}Please enter a valid number.{Style.RESET_ALL}")
        
        # Max drawdown
        while True:
            try:
                max_drawdown = float(input(f"\n{Fore.GREEN}Enter maximum allowed drawdown (0-1): {Style.RESET_ALL}").strip())
                if 0 < max_drawdown <= 1:
                    params['max_drawdown'] = max_drawdown
                    break
                print(f"{Fore.RED}Value must be between 0 and 1.{Style.RESET_ALL}")
            except ValueError:
                print(f"{Fore.RED}Please enter a valid number.{Style.RESET_ALL}")
        
        # Risk per trade
        while True:
            try:
                risk = float(input(f"\n{Fore.GREEN}Enter risk per trade (0-1): {Style.RESET_ALL}").strip())
                if 0 < risk <= 1:
                    params['risk_per_trade'] = risk
                    break
                print(f"{Fore.RED}Value must be between 0 and 1.{Style.RESET_ALL}")
            except ValueError:
                print(f"{Fore.RED}Please enter a valid number.{Style.RESET_ALL}")
        
        return params
    
    def _configure_performance_metrics_parameters(self) -> Optional[Dict]:
        """Configure performance metrics parameters."""
        print(f"\n{Fore.CYAN}=== Performance Metrics Parameters ==={Style.RESET_ALL}")
        
        params = {}
        
        # Benchmark symbol
        use_benchmark = self._get_boolean_input("Use benchmark for comparison? (y/n): ")
        if use_benchmark:
            benchmark = input(f"\n{Fore.GREEN}Enter benchmark symbol: {Style.RESET_ALL}").strip().upper()
            params['benchmark_symbol'] = benchmark
        
        # Risk-free rate
        while True:
            try:
                rate = float(input(f"\n{Fore.GREEN}Enter risk-free rate (0-1): {Style.RESET_ALL}").strip())
                if 0 <= rate <= 1:
                    params['risk_free_rate'] = rate
                    break
                print(f"{Fore.RED}Value must be between 0 and 1.{Style.RESET_ALL}")
            except ValueError:
                print(f"{Fore.RED}Please enter a valid number.{Style.RESET_ALL}")
        
        return params
    
    def _configure_data_validation_parameters(self) -> Optional[Dict]:
        """Configure data validation parameters."""
        print(f"\n{Fore.CYAN}=== Data Validation Parameters ==={Style.RESET_ALL}")
        
        params = {}
        
        # Data validation
        params['data_validation'] = self._get_boolean_input("Enable data validation? (y/n): ")
        
        if params['data_validation']:
            # Required columns
            print(f"\n{Fore.GREEN}Enter required columns (comma-separated):{Style.RESET_ALL}")
            print("Example: open,high,low,close,volume")
            columns = input().strip().split(',')
            params['required_columns'] = [col.strip() for col in columns if col.strip()]
        
        return params
    
    def _get_boolean_input(self, prompt: str, allow_skip: bool = False) -> Optional[bool]:
        """Get boolean input from user."""
        while True:
            response = input(f"\n{Fore.GREEN}{prompt}{Style.RESET_ALL}").strip().lower()
            if allow_skip and not response:  # Allow skipping if enabled
                return None
            if response in ['y', 'yes', '1', 'true']:
                return True
            if response in ['n', 'no', '0', 'false']:
                return False
            print(f"{Fore.RED}Please enter 'y' or 'n'.{Style.RESET_ALL}")
    
    def _serialize_results(self, results: Dict) -> Dict:
        """Serialize results to ensure JSON compatibility."""
        try:
            # Convert the results to a JSON string and back to handle all serialization
            return json.loads(json.dumps(results, cls=DateTimeEncoder))
        except Exception as e:
            logger.error(f"Error serializing results: {str(e)}")
            # Return a simplified version of the results
            return {
                'error': str(e),
                'message': 'Error serializing results'
            }

    def _load_saved_configs(self, config_type: str = 'backtest') -> List[Dict]:
        """Load saved configurations from JSON files.
        
        Args:
            config_type (str): Type of configurations to load ('backtest', 'forward_test', or 'optimization')
            
        Returns:
            List[Dict]: List of saved configurations
        """
        try:
            config_dir = os.path.join(
                os.path.dirname(__file__),
                'results',
                config_type
            )
            
            if not os.path.exists(config_dir):
                logger.warning(f"No saved configurations found in {config_dir}")
                return []
            
            # Load config.json
            config_file = os.path.join(config_dir, 'config.json')
            if not os.path.exists(config_file):
                return []
            
            with open(config_file, 'r') as f:
                configs = json.load(f)
                
                # Handle empty optimization config
                if config_type == 'optimization' and configs == []:
                    return []
                
                # Convert dict of configs to list
                if isinstance(configs, dict):
                    config_list = []
                    for test_id, config in configs.items():
                        # Create a flattened configuration with test_id
                        flat_config = {
                            'test_id': test_id,
                            **config
                        }
                        config_list.append(flat_config)
                    return config_list
                
                return configs
            
        except Exception as e:
            logger.error(f"Error loading saved configurations: {str(e)}")
            return []

    def _select_saved_config(self, config_type: str = 'backtest') -> Optional[Dict]:
        """Display saved configurations and let user select one.
        
        Args:
            config_type (str): Type of configurations to display ('backtest', 'forward_test', or 'optimization')
            
        Returns:
            Optional[Dict]: Selected configuration if any, None otherwise
        """
        configs = self._load_saved_configs(config_type)
        
        if not configs:
            print(f"\n{Fore.YELLOW}No saved configurations found.{Style.RESET_ALL}")
            return None
        
        # Sort configs by timestamp
        configs.sort(key=lambda x: x.get('metadata', {}).get('timestamp', ''), reverse=True)
        
        # Display configs
        print(f"\n{Fore.CYAN}=== Saved Configurations ==={Style.RESET_ALL}")
        options = []
        
        for i, config in enumerate(configs, 1):
            # Get metadata from config
            metadata = config.get('metadata', {})
            timestamp = metadata.get('timestamp', 'Unknown')
            ticker = metadata.get('ticker', 'Unknown')
            timeframe = metadata.get('timeframe', 'Unknown')
            strategy = metadata.get('strategy_name', 'Unknown')
            
            # Get test ID
            test_id = config.get('test_id', 'Unknown')
            
            # Get summary based on config type
            if config_type == 'backtest':
                data = config.get('data', {})
                backtesting_params = data.get('backtesting_params', {})
                forward_test = backtesting_params.get('forward_test', False)
                summary = f"Forward Test: {'Yes' if forward_test else 'No'}"
            elif config_type == 'forward_test':
                data = config.get('data', {})
                scenarios = data.get('scenarios', [])
                summary = f"Scenarios: {', '.join(scenarios)}"
            else:  # optimization
                summary = "Optimization Configuration"
            
            print(f"\n{Fore.GREEN}{i}. Test ID: {test_id}")
            print(f"   Timestamp: {timestamp}")
            print(f"   Ticker: {ticker}")
            print(f"   Timeframe: {timeframe}")
            print(f"   Strategy: {strategy}")
            print(f"   {summary}")
            
            options.append((test_id, f"{timestamp} - {ticker} - {strategy}"))
        
        print("\n0. Back")
        
        while True:
            try:
                choice = int(input(f"\n{Fore.GREEN}Select configuration (0-{len(options)}): {Style.RESET_ALL}").strip())
                if choice == 0:
                    return None
                if 1 <= choice <= len(options):
                    selected_config = self._load_config_by_id(options[choice-1][0], config_type)
                    if selected_config:
                        print(f"\n{Fore.GREEN}Configuration loaded successfully!{Style.RESET_ALL}")
                        return selected_config
                    else:
                        print(f"\n{Fore.RED}Error loading configuration.{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}Invalid choice!{Style.RESET_ALL}")
            except ValueError:
                print(f"{Fore.RED}Please enter a valid number.{Style.RESET_ALL}")

    def _load_config_by_id(self, test_id: str, config_type: str = 'backtest') -> Optional[Dict]:
        """Load specific configuration by test ID.
        
        Args:
            test_id (str): Test ID to load
            config_type (str): Type of configuration to load ('backtest', 'forward_test', or 'optimization')
            
        Returns:
            Optional[Dict]: Configuration if found, None otherwise
        """
        try:
            config_dir = os.path.join(
                os.path.dirname(__file__),
                'results',
                config_type
            )
            
            config_file = os.path.join(config_dir, 'config.json')
            if not os.path.exists(config_file):
                return None
            
            with open(config_file, 'r') as f:
                configs = json.load(f)
                
                # Handle empty optimization config
                if config_type == 'optimization' and configs == []:
                    return None
                
                # Handle dict of configs
                if isinstance(configs, dict):
                    if test_id in configs:
                        return configs[test_id]
                
                # Handle list of configs
                elif isinstance(configs, list):
                    for config in configs:
                        if config.get('test_id') == test_id:
                            return config
            
            return None
            
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return None
    
    def _backtest_flow(self):
        """Handle the backtesting flow."""
        try:
            # Ask if user wants to load saved config
            load_saved = self._get_boolean_input("Load saved configuration? (y/n): ")
            
            if load_saved:
                saved_config = self._select_saved_config('backtest')
                if saved_config:
                    # Use saved configuration
                    self._use_saved_config(saved_config)
                    return
            
            # Continue with normal flow if no saved config or user chose not to load
            # 1. Select Strategy
            strategy_options = [(name, name.replace('_', ' ').title()) 
                              for name in self.engine.get_available_strategies().keys()]
            strategy_name = self._display_menu("Select Strategy", strategy_options)
            if strategy_name == 'back':
                return
            
            # 2. Get Instrument Key
            instrument_key = self._get_instrument_key()
            if instrument_key is None:
                return
            
            # 3. Configure Data Parameters
            data_params = self._configure_data_parameters()
            if data_params is None:
                return
            
            # 4. Load Data
            print(f"\n{Fore.CYAN}Loading data...{Style.RESET_ALL}")
            data = self.engine.load_data(
                instrument_key=instrument_key,
                unit=data_params['unit'],
                interval=data_params['interval'],
                data_type=data_params['data_type'],
                days=data_params.get('days', 0)  # Only used for historical data
            )
            
            # 5. Configure Strategy Parameters
            strategy_params = self._configure_strategy_parameters(strategy_name)
            if strategy_params is None:
                return
            
            # 6. Configure Basic Tester Parameters
            tester_params = self._configure_tester_parameters()
            if tester_params is None:
                return
            
            # 7. Configure Additional Parameters (Optional)
            print(f"\n{Fore.YELLOW}=== Optional Parameters ==={Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Press Enter to skip optional parameters{Style.RESET_ALL}")
            
            # Ask if user wants to configure additional parameters
            configure_additional = self._get_boolean_input("Configure additional parameters? (y/n): ", allow_skip=True)
            
            additional_params = {}
            if configure_additional:
                # Configure each parameter group
                param_groups = {
                    'Backtesting': self._configure_backtesting_parameters,
                    'Backward Testing': self._configure_backward_testing_parameters,
                    'Statistical': self._configure_statistical_parameters,
                    'Trade Engine': self._configure_trade_engine_parameters,
                    'Risk Management': self._configure_risk_management_parameters,
                    'Performance Metrics': self._configure_performance_metrics_parameters,
                    'Data Validation': self._configure_data_validation_parameters
                }
                
                for group_name, config_func in param_groups.items():
                    print(f"\n{Fore.CYAN}=== {group_name} Parameters ==={Style.RESET_ALL}")
                    params = config_func()
                    if params is not None:
                        additional_params.update(params)
            
            # 8. Run Strategy
            print(f"\n{Fore.CYAN}Running strategy...{Style.RESET_ALL}")
            try:
                # Prepare strategy parameters
                strategy_config = {
                    'data': data,
                    'strategy_name': strategy_name,
                    'strategy_params': strategy_params,
                    'tester_params': {
                        'ticker': instrument_key,
                        'timeframe': f"{data_params['interval']}{data_params['unit']}",
                        'testing_period_months': data_params.get('days', 0) // 30
                    }
                }
                
                # Add additional parameters if configured
                if additional_params:
                    strategy_config.update(additional_params)
                
                # Ensure all numeric parameters are properly converted
                for param_name, param_value in strategy_config.get('strategy_params', {}).items():
                    if isinstance(param_value, str) and param_value.replace('.', '').isdigit():
                        strategy_config['strategy_params'][param_name] = float(param_value)
                
                results = self.engine.run_strategy(**strategy_config)
                
            except Exception as e:
                logger.error(f"Error running strategy: {str(e)}")
                print(f"\n{Fore.RED}Error running strategy: {str(e)}{Style.RESET_ALL}")
                return
            
            # Serialize results before processing
            serialized_results = self._serialize_results(results)
            
            # 9. Display Results
            print(f"\n{Fore.GREEN}Strategy Test Results:{Style.RESET_ALL}")
            print("-" * 50)
            
            if 'trades' in serialized_results:
                print(f"\nTotal Trades: {len(serialized_results['trades'])}")
            
            if 'metrics' in serialized_results:
                print("\nPerformance Metrics:")
                metrics_df = pd.DataFrame(serialized_results['metrics'])
                print(tabulate(metrics_df, headers='keys', tablefmt='psql'))
            
            if 'statistics' in serialized_results:
                print("\nStatistics:")
                stats_df = pd.DataFrame(serialized_results['statistics'])
                print(tabulate(stats_df, headers='keys', tablefmt='psql'))
            
            if 'recommendation' in serialized_results:
                print(f"\nRecommendation: {serialized_results['recommendation']}")
            
        except Exception as e:
            logger.error(f"Error in backtest flow: {str(e)}")
            print(f"\n{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
    
    def _optimization_flow(self):
        """Handle the optimization flow."""
        try:
            # Ask if user wants to load saved config
            load_saved = self._get_boolean_input("Load saved configuration? (y/n): ")
            
            if load_saved:
                saved_config = self._select_saved_config('optimization')
                if saved_config:
                    # Use saved configuration
                    self._use_saved_config(saved_config)
                    return
            
            # Continue with normal flow if no saved config or user chose not to load
            # 1. Select Strategy
            strategy_options = [(name, name.replace('_', ' ').title()) 
                              for name in self.engine.get_available_strategies().keys()]
            strategy_name = self._display_menu("Select Strategy", strategy_options)
            if strategy_name == 'back':
                return
            
            # 2. Get Instrument Key
            instrument_key = self._get_instrument_key()
            if instrument_key is None:
                return
            
            # 3. Configure Data Parameters
            data_params = self._configure_data_parameters()
            if data_params is None:
                return
            
            # 4. Load Data
            print(f"\n{Fore.CYAN}Loading data...{Style.RESET_ALL}")
            data = self.engine.load_data(
                instrument_key=instrument_key,
                unit=data_params['unit'],
                interval=data_params['interval'],
                data_type=data_params['data_type'],
                days=data_params.get('days', 0)  # Only used for historical data
            )
            
            # 5. Configure Strategy Parameters
            strategy_params = self._configure_strategy_parameters(strategy_name)
            if strategy_params is None:
                return
            
            # 6. Configure Optimizer Parameters
            optimizer_params = self._configure_optimizer_parameters()
            if optimizer_params is None:
                return
            
            # 7. Configure Additional Parameters (Optional)
            print(f"\n{Fore.YELLOW}=== Optional Parameters ==={Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Press Enter to skip optional parameters{Style.RESET_ALL}")
            
            # Ask if user wants to configure additional parameters
            configure_additional = self._get_boolean_input("Configure additional parameters? (y/n): ", allow_skip=True)
            
            additional_params = {}
            if configure_additional:
                # Configure each parameter group
                param_groups = {
                    'Backtesting': self._configure_backtesting_parameters,
                    'Backward Testing': self._configure_backward_testing_parameters,
                    'Statistical': self._configure_statistical_parameters,
                    'Trade Engine': self._configure_trade_engine_parameters,
                    'Risk Management': self._configure_risk_management_parameters,
                    'Performance Metrics': self._configure_performance_metrics_parameters,
                    'Data Validation': self._configure_data_validation_parameters
                }
                
                for group_name, config_func in param_groups.items():
                    print(f"\n{Fore.CYAN}=== {group_name} Parameters ==={Style.RESET_ALL}")
                    params = config_func()
                    if params is not None:
                        additional_params.update(params)
            
            # 8. Run Optimization
            print(f"\n{Fore.CYAN}Running optimization...{Style.RESET_ALL}")
            try:
                # Get strategy class from engine
                strategy_class = self.engine.get_available_strategies()[strategy_name]['class']
                
                # Prepare parameter bounds from strategy parameters
                parameter_bounds = []
                for param_name, param_info in self.engine.get_available_strategies()[strategy_name]['params'].items():
                    if param_name in strategy_params:
                        # Create bounds around the parameter value
                        value = strategy_params[param_name]
                        if isinstance(value, (int, float)):
                            lower = max(1, int(value * 0.5))
                            upper = int(value * 1.5)
                            parameter_bounds.append((lower, upper))
                
                # Prepare optimization parameters
                opt_params = {
                    'ticker': instrument_key,
                    'strategy_class': strategy_class,
                    'parameter_bounds': parameter_bounds,
                    'data': data,
                    'criterion': optimizer_params['criterion'],
                    'timeframe': f"{data_params['interval']}{data_params['unit']}",
                    'trade_engine_type': optimizer_params.get('trade_engine_type', 'Signals'),
                    'optimizing_period': data_params.get('days', 0) // 30,  # Convert days to months
                    'balance': additional_params.get('investing_amount', 1000.0),
                    'leverage': additional_params.get('account_leverage', 500.0),
                    'volume': additional_params.get('volume', 1),
                    'commission': additional_params.get('commission', 7.0),
                    'optimization_method': optimizer_params['method'],
                    'max_combinations': optimizer_params['max_combinations']
                }

                # Store the configuration with proper days value
                # Generate unique test ID with strategy name and symbol
                strategy_name = strategy_name.replace('_', ' ').title()
                symbol = instrument_key.split('|')[0] if '|' in instrument_key else instrument_key
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                test_id = f"{strategy_name}_{symbol}_{timestamp}"

                # Calculate days from timeframe if not provided
                days = data_params.get('days', 0)
                if days == 0:
                    # Calculate days based on timeframe
                    interval = data_params['interval']
                    unit = data_params['unit']
                    if unit == 'minutes':
                        days = interval * 1440  # Convert to days
                    elif unit == 'hours':
                        days = interval * 24
                else:
                        days = interval

                # Get strategy parameters with names
                strategy_info = self.engine.get_available_strategies()[strategy_name]
                named_parameter_bounds = {}
                for param_name, param_info in strategy_info['params'].items():
                    if param_name in strategy_params:
                        value = strategy_params[param_name]
                        if isinstance(value, (int, float)):
                            lower = max(1, int(value * 0.5))
                            upper = int(value * 1.5)
                            named_parameter_bounds[param_name] = [lower, upper]

                config = {
                    test_id: {
                        'data': {
                            'data_loading_params': {
                                'instrument_key': instrument_key,
                                'unit': data_params['unit'],
                                'interval': data_params['interval'],
                                'data_type': data_params['data_type'],
                                'days': days  # Store the calculated days value
                            },
                            'strategy_params': {
                                'parameters': strategy_params,  # Store actual strategy parameters
                                'parameter_bounds': named_parameter_bounds,  # Store named parameter bounds
                                'strategy_name': strategy_name  # Add strategy name for reference
                            },
                            'optimization_params': {
                                'criterion': optimizer_params['criterion'],
                                'optimization_method': optimizer_params['method'],
                                'max_combinations': optimizer_params['max_combinations']
                            },
                            'backtesting_params': {
                                'trade_engine_type': opt_params['trade_engine_type'],
                                'investing_amount': opt_params['balance'],
                                'account_leverage': opt_params['leverage'],
                                'volume': opt_params['volume'],
                                'commission': opt_params['commission']
                            },
                            'data_validation_params': {
                                'required_columns': ['time', 'open', 'high', 'low', 'close', 'volume'],
                                'data_validation': True
                            }
                        },
                        'metadata': {
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'ticker': instrument_key,
                            'timeframe': opt_params['timeframe'],
                            'strategy_name': strategy_name
                        }
                    }
                }
                
                # Store the configuration
                output_dir = os.path.join(
                    os.path.dirname(__file__),
                    'results',
                    'optimization'
                )
                os.makedirs(output_dir, exist_ok=True)
                
                config_file = os.path.join(output_dir, 'config.json')
                
                # Read existing config if it exists
                if os.path.exists(config_file):
                    with open(config_file, 'r') as f:
                        try:
                            existing_config = json.load(f)
                            if isinstance(existing_config, dict):
                                existing_config.update(config)
                                config = existing_config
                        except json.JSONDecodeError:
                            pass
                
                # Write the updated config
                with open(config_file, 'w') as f:
                    json.dump(config, f, indent=4)
                
                results = self.engine.optimize_strategy(**opt_params)
                
            except Exception as e:
                logger.error(f"Error running optimization: {str(e)}")
                print(f"\n{Fore.RED}Error running optimization: {str(e)}{Style.RESET_ALL}")
                return
            
            # Serialize results before processing
            serialized_results = self._serialize_results(results)
            
            # 9. Display Results
            print(f"\n{Fore.GREEN}Optimization Results:{Style.RESET_ALL}")
            print("-" * 50)
            
            if 'best_params' in serialized_results:
                print("\nBest Parameters:")
                if isinstance(serialized_results['best_params'], dict):
                    for param, value in serialized_results['best_params'].items():
                        print(f"{param}: {value}")
                else:
                    print(serialized_results['best_params'])
            
            if 'performance' in serialized_results:
                print("\nPerformance Metrics:")
                if isinstance(serialized_results['performance'], list) and serialized_results['performance']:
                    perf_df = pd.DataFrame(serialized_results['performance'])
                    print(tabulate(perf_df, headers='keys', tablefmt='psql'))
                else:
                    print("No performance metrics available")
            
            if 'statistics' in serialized_results:
                print("\nStatistics:")
                if isinstance(serialized_results['statistics'], dict) and serialized_results['statistics']:
                    stats_df = pd.DataFrame(serialized_results['statistics'])
                    print(tabulate(stats_df, headers='keys', tablefmt='psql'))
                else:
                    print("No statistics available")
            
            if 'recommendation' in serialized_results:
                print(f"\nRecommendation: {serialized_results['recommendation']}")
            
        except Exception as e:
            logger.error(f"Error in optimization flow: {str(e)}")
            print(f"\n{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
    
    def _use_saved_config(self, config: Dict) -> None:
        """Use a saved configuration for testing or optimization.
        
        Args:
            config (Dict): Configuration to use
        """
        try:
            # Extract data from the nested structure
            data = config.get('data', {})
            metadata = config.get('metadata', {})
            
            # Extract all parameters from config first
            data_loading_params = data.get('data_loading_params', {})
            strategy_params = data.get('strategy_params', {})
            backtesting_params = data.get('backtesting_params', {})
            backward_testing_params = data.get('backward_testing_params', {})
            statistical_params = data.get('statistical_params', {})
            trade_engine_params = data.get('trade_engine_params', {})
            risk_management_params = data.get('risk_management_params', {})
            performance_metrics_params = data.get('performance_metrics_params', {})
            data_validation_params = data.get('data_validation_params', {})
            optimization_params = data.get('optimization_params', {})
            
            # Check if this is an optimization config
            is_optimization = bool(optimization_params)
            
            if not data_loading_params:
                raise ValueError("Data loading parameters not found in configuration")
            
            # Get strategy name and load the strategy class
            strategy_name = metadata.get('strategy_name')
            if not strategy_name:
                raise ValueError("Strategy name not found in configuration")
            
            # Convert strategy name to match engine's format
            if strategy_name == 'MovingAverageCrossoverStrategy':
                strategy_name = 'moving_average'
            elif strategy_name == 'RSIStrategy':
                strategy_name = 'rsi'
            
            # Get available strategies from engine
            available_strategies = self.engine.get_available_strategies()
            if strategy_name not in available_strategies:
                raise ValueError(f"Strategy {strategy_name} not found in available strategies")
            
            # Get the strategy class
            strategy_class = available_strategies[strategy_name]['class']
            
            # Calculate days from timeframe if needed
            days = data_loading_params.get('days', 0)
            if days == 0:
                timeframe = metadata.get('timeframe', '')
                if timeframe:
                    try:
                        interval = int(''.join(filter(str.isdigit, timeframe)))
                        unit = ''.join(filter(str.isalpha, timeframe))
                        if unit == 'minutes':
                            days = interval * 1440
                        elif unit == 'hours':
                            days = interval * 24
                        else:
                            days = interval
                    except ValueError:
                        days = 30
            
            # Load data with proper parameters
            print(f"\n{Fore.CYAN}Loading data...{Style.RESET_ALL}")
            try:
                data = self.engine.load_data(
                    instrument_key=data_loading_params['instrument_key'],
                    unit=data_loading_params['unit'],
                    interval=data_loading_params['interval'],
                    data_type=data_loading_params['data_type'],
                    days=days
                )
            except Exception as e:
                logger.error(f"Error loading data: {str(e)}")
                raise ValueError(f"Failed to load data: {str(e)}")
            
            if is_optimization:
                # Get parameter bounds from strategy parameters
                parameter_bounds = strategy_params.get('parameter_bounds', [])
                
                # If parameter bounds is a list of lists, use it directly
                if isinstance(parameter_bounds, list) and all(isinstance(bound, list) for bound in parameter_bounds):
                    # Convert to list of tuples for optimization
                    parameter_bounds = [tuple(bound) for bound in parameter_bounds]
                else:
                    # If no parameter bounds found or in wrong format, generate them from strategy parameters
                    strategy_info = self.engine.get_available_strategies()[strategy_name]
                    parameter_bounds = []
                    for param_name, param_info in strategy_info['params'].items():
                        if isinstance(param_info.get('default'), (int, float)):
                            default_value = param_info['default']
                            lower = max(1, int(default_value * 0.5))
                            upper = int(default_value * 1.5)
                            parameter_bounds.append((lower, upper))
                
                if not parameter_bounds:
                    raise ValueError("No parameter bounds found in configuration and could not generate default bounds")
                
                # Run optimization
                print(f"\n{Fore.CYAN}Running optimization...{Style.RESET_ALL}")
                try:
                    results = self.engine.optimize_strategy(
                        ticker=data_loading_params['instrument_key'],
                        strategy_class=strategy_class,
                        parameter_bounds=parameter_bounds,
                        data=data,
                        criterion=optimization_params.get('criterion', 'Balance_Max'),
                        timeframe=f"{data_loading_params['interval']}{data_loading_params['unit']}",
                        trade_engine_type=backtesting_params.get('trade_engine_type', 'Signals'),
                        optimizing_period=days // 30,  # Convert days to months
                        balance=backtesting_params.get('investing_amount', 1000.0),
                        leverage=backtesting_params.get('account_leverage', 500.0),
                        volume=backtesting_params.get('volume', 1),
                        commission=backtesting_params.get('commission', 7.0),
                        optimization_method=optimization_params.get('optimization_method', 'grid_search'),
                        max_combinations=optimization_params.get('max_combinations', 1000)
                    )
                except Exception as e:
                    logger.error(f"Error running optimization: {str(e)}")
                    raise ValueError(f"Failed to run optimization: {str(e)}")
            else:
                # Run strategy
                print(f"\n{Fore.CYAN}Running strategy with loaded configuration...{Style.RESET_ALL}")
                try:
                    results = self.engine.run_strategy(
                        data=data,
                        strategy_name=strategy_name,
                        strategy_params=strategy_params,
                        tester_params={
                            'ticker': data_loading_params['instrument_key'],
                            'timeframe': f"{data_loading_params['interval']}{data_loading_params['unit']}",
                            'testing_period_months': days // 30
                        },
                        # Backtesting parameters
                        forward_test=backtesting_params.get('forward_test', True),
                        trade_engine_type=backtesting_params.get('trade_engine_type', 'Signals'),
                        investing_amount=backtesting_params.get('investing_amount', 1000.0),
                        account_leverage=backtesting_params.get('account_leverage', 500.0),
                        volume=backtesting_params.get('volume', 1),
                        commission=backtesting_params.get('commission', 7.0),
                        base_price=backtesting_params.get('base_price', 'close'),
                        # Backward testing parameters
                        separate_close_signals=backward_testing_params.get('separate_close_signals', False),
                        parallel_opening=backward_testing_params.get('parallel_opening', False),
                        max_open_positions=backward_testing_params.get('max_open_positions', 1),
                        position_sizing_method=backward_testing_params.get('position_sizing_method', 'fixed'),
                        allow_shorting=backward_testing_params.get('allow_shorting', True),
                        allow_partial_fills=backward_testing_params.get('allow_partial_fills', False),
                        slippage_model=backward_testing_params.get('slippage_model', 'none'),
                        slippage_value=backward_testing_params.get('slippage_value', 0.0),
                        # Statistical testing parameters
                        significance_level=statistical_params.get('significance_level', 0.05),
                        monte_carlo_simulations=statistical_params.get('monte_carlo_simulations', 1000),
                        confidence_level=statistical_params.get('confidence_level', 0.95),
                        # Trade engine parameters
                        stop_loss_pct=trade_engine_params.get('stop_loss_pct', 2.0),
                        take_profit_pct=trade_engine_params.get('take_profit_pct', 4.0),
                        trailing_stop=trade_engine_params.get('trailing_stop', False),
                        trailing_stop_pct=trade_engine_params.get('trailing_stop_pct', 1.0),
                        # Risk management parameters
                        max_position_size=risk_management_params.get('max_position_size', 0.1),
                        max_drawdown=risk_management_params.get('max_drawdown', 0.2),
                        risk_per_trade=risk_management_params.get('risk_per_trade', 0.02),
                        # Performance metrics parameters
                        benchmark_symbol=performance_metrics_params.get('benchmark_symbol'),
                        risk_free_rate=performance_metrics_params.get('risk_free_rate', 0.05),
                        # Data validation parameters
                        required_columns=data_validation_params.get('required_columns', ['time', 'open', 'high', 'low', 'close', 'volume']),
                        data_validation=data_validation_params.get('data_validation', True)
                    )
                except Exception as e:
                    logger.error(f"Error running strategy: {str(e)}")
                    raise ValueError(f"Failed to run strategy: {str(e)}")
            
            # Display results
            print(f"\n{Fore.GREEN}{'Optimization' if is_optimization else 'Strategy Test'} Results:{Style.RESET_ALL}")
            print("-" * 50)
            
            if is_optimization:
                if 'best_params' in results:
                    print("\nBest Parameters:")
                    if isinstance(results['best_params'], dict):
                        for param, value in results['best_params'].items():
                            print(f"{param}: {value}")
                    else:
                        print(results['best_params'])
                
                if 'performance' in results:
                    print("\nPerformance Metrics:")
                    if isinstance(results['performance'], list) and results['performance']:
                        perf_df = pd.DataFrame(results['performance'])
                        print(tabulate(perf_df, headers='keys', tablefmt='psql'))
                    else:
                        print("No performance metrics available")
            else:
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
            
        except ValueError as e:
            logger.error(f"Configuration error: {str(e)}")
            print(f"\n{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        except Exception as e:
            logger.error(f"Error using saved configuration: {str(e)}")
            print(f"\n{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
    
    def run(self):
        """Run the CLI main loop."""
        try:
            while True:
                # Main menu
                options = [
                    ('backtest', 'Run Backtest'),
                    ('optimize', 'Run Optimization'),
                    ('exit', 'Exit')
                ]
                choice = self._display_menu("Strategy Testing and Optimization CLI", options, allow_back=False)
                
                if choice == 'exit':
                    print(f"\n{Fore.GREEN}Exiting...{Style.RESET_ALL}")
                    break
                
                # Handle menu choices
                if choice == 'backtest':
                    self._backtest_flow()
                elif choice == 'optimize':
                    self._optimization_flow()
                
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Program terminated by user.{Style.RESET_ALL}")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            print(f"\n{Fore.RED}An unexpected error occurred: {str(e)}{Style.RESET_ALL}")
        finally:
            print(f"\n{Fore.GREEN}Thank you for using Strategy Testing and Optimization CLI!{Style.RESET_ALL}")

def main():
    """Main entry point."""
    cli = StrategyCLI()
    cli.run()

if __name__ == "__main__":
    main() 