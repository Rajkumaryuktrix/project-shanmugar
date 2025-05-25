import logging
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime
from colorama import Fore, Style
from tabulate import tabulate

from src.strategy_module.core_engine import StrategyEngine
from src.strategy_module.utils.instrument_finder import InstrumentKeyFinder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    
    def _display_menu(self, title: str, options: List[tuple], allow_back: bool = True) -> str:
        """Display a menu and get user choice."""
        print(f"\n{Fore.CYAN}=== {title} ==={Style.RESET_ALL}")
        
        for i, (key, label) in enumerate(options, 1):
            print(f"{i}. {label}")
        
        if allow_back:
            print("0. Back")
        
        while True:
            try:
                choice = int(input(f"\n{Fore.GREEN}Enter your choice: {Style.RESET_ALL}").strip())
                if choice == 0 and allow_back:
                    return 'back'
                if 1 <= choice <= len(options):
                    return options[choice - 1][0]
                print(f"{Fore.RED}Invalid choice!{Style.RESET_ALL}")
            except ValueError:
                print(f"{Fore.RED}Please enter a valid number.{Style.RESET_ALL}")
    
    def _get_instrument_key(self) -> Optional[str]:
        """Get instrument key from user input."""
        while True:
            symbol = input(f"\n{Fore.GREEN}Enter trading symbol (e.g., RELIANCE): {Style.RESET_ALL}").strip().upper()
            try:
                instrument_key = self.instrument_finder.get_instrument_key(symbol)
                if instrument_key:
                    return instrument_key
                print(f"{Fore.RED}Symbol not found!{Style.RESET_ALL}")
            except Exception as e:
                logger.error(f"Error validating symbol: {str(e)}")
                print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
    
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
                    
                    params[param_name] = value
                    break
                except ValueError:
                    print(f"{Fore.RED}Invalid value. Please enter a valid {param_info['type']}.{Style.RESET_ALL}")
        
        return params
    
    def _configure_tester_parameters(self) -> Optional[Dict]:
        """Configure tester parameters with interactive menu."""
        print(f"\n{Fore.CYAN}=== Tester Parameters ==={Style.RESET_ALL}")
        
        params = {}
        
        # Testing period
        while True:
            try:
                testing_period = int(input(f"\n{Fore.GREEN}Enter testing period in months (e.g., 3): {Style.RESET_ALL}").strip())
                if testing_period > 0:
                    params['testing_period_months'] = testing_period
                    break
                print(f"{Fore.RED}Period must be greater than 0.{Style.RESET_ALL}")
            except ValueError:
                print(f"{Fore.RED}Please enter a valid number.{Style.RESET_ALL}")
        
        # Investment amount
        while True:
            try:
                investment = float(input(f"\n{Fore.GREEN}Enter investment amount (e.g., 1000): {Style.RESET_ALL}").strip())
                if investment > 0:
                    params['investing_amount'] = investment
                    break
                print(f"{Fore.RED}Amount must be greater than 0.{Style.RESET_ALL}")
            except ValueError:
                print(f"{Fore.RED}Please enter a valid number.{Style.RESET_ALL}")
        
        # Account leverage
        while True:
            try:
                leverage = float(input(f"\n{Fore.GREEN}Enter account leverage (e.g., 500): {Style.RESET_ALL}").strip())
                if leverage > 0:
                    params['account_leverage'] = leverage
                    break
                print(f"{Fore.RED}Leverage must be greater than 0.{Style.RESET_ALL}")
            except ValueError:
                print(f"{Fore.RED}Please enter a valid number.{Style.RESET_ALL}")
        
        # Trading volume
        while True:
            try:
                volume = float(input(f"\n{Fore.GREEN}Enter trading volume (e.g., 0.01): {Style.RESET_ALL}").strip())
                if 0 < volume <= 1:
                    params['volume'] = volume
                    break
                print(f"{Fore.RED}Volume must be between 0 and 1.{Style.RESET_ALL}")
            except ValueError:
                print(f"{Fore.RED}Please enter a valid number.{Style.RESET_ALL}")
        
        # Commission
        while True:
            try:
                commission = float(input(f"\n{Fore.GREEN}Enter commission percentage (e.g., 7.0): {Style.RESET_ALL}").strip())
                if commission >= 0:
                    params['commission'] = commission
                    break
                print(f"{Fore.RED}Commission must be non-negative.{Style.RESET_ALL}")
            except ValueError:
                print(f"{Fore.RED}Please enter a valid number.{Style.RESET_ALL}")
        
        # Stop loss and take profit
        while True:
            try:
                stop_loss = float(input(f"\n{Fore.GREEN}Enter stop loss percentage (e.g., 2.0): {Style.RESET_ALL}").strip())
                take_profit = float(input(f"{Fore.GREEN}Enter take profit percentage (e.g., 4.0): {Style.RESET_ALL}").strip())
                if 0 < stop_loss < take_profit:
                    params['stop_loss_pct'] = stop_loss
                    params['take_profit_pct'] = take_profit
                    break
                print(f"{Fore.RED}Stop loss must be less than take profit.{Style.RESET_ALL}")
            except ValueError:
                print(f"{Fore.RED}Please enter valid numbers.{Style.RESET_ALL}")
        
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
        
        # Forward test
        forward_test = self._get_boolean_input("Run forward testing? (y/n): ")
        params['forward_test'] = forward_test
        
        # Trade engine type
        engine_options = [
            ('Signals', 'Signal-based Trading'),
            ('SLTP', 'Stop Loss/Take Profit Trading')
        ]
        engine_type = self._display_menu("Select Trade Engine Type", engine_options)
        if engine_type == 'back':
            return None
        params['trade_engine_type'] = engine_type
        
        # Base price
        price_options = [
            ('open', 'Open Price'),
            ('high', 'High Price'),
            ('low', 'Low Price'),
            ('close', 'Close Price')
        ]
        base_price = self._display_menu("Select Base Price", price_options)
        if base_price == 'back':
            return None
        params['base_price'] = base_price
        
        return params
    
    def _configure_backward_testing_parameters(self) -> Optional[Dict]:
        """Configure backward testing specific parameters."""
        print(f"\n{Fore.CYAN}=== Backward Testing Parameters ==={Style.RESET_ALL}")
        
        params = {}
        
        # Separate close signals
        params['separate_close_signals'] = self._get_boolean_input("Handle close signals separately? (y/n): ")
        
        # Parallel opening
        params['parallel_opening'] = self._get_boolean_input("Allow parallel position opening? (y/n): ")
        
        # Max open positions
        while True:
            try:
                max_positions = int(input(f"\n{Fore.GREEN}Enter maximum number of open positions: {Style.RESET_ALL}").strip())
                if max_positions > 0:
                    params['max_open_positions'] = max_positions
                    break
                print(f"{Fore.RED}Number must be greater than 0.{Style.RESET_ALL}")
            except ValueError:
                print(f"{Fore.RED}Please enter a valid number.{Style.RESET_ALL}")
        
        # Position sizing method
        sizing_options = [
            ('fixed', 'Fixed Size'),
            ('risk_based', 'Risk-based'),
            ('kelly', 'Kelly Criterion')
        ]
        sizing_method = self._display_menu("Select Position Sizing Method", sizing_options)
        if sizing_method == 'back':
            return None
        params['position_sizing_method'] = sizing_method
        
        # Allow shorting
        params['allow_shorting'] = self._get_boolean_input("Allow short positions? (y/n): ")
        
        # Allow partial fills
        params['allow_partial_fills'] = self._get_boolean_input("Allow partial order fills? (y/n): ")
        
        # Slippage model
        slippage_options = [
            ('none', 'No Slippage'),
            ('fixed', 'Fixed Slippage'),
            ('percentage', 'Percentage Slippage'),
            ('random', 'Random Slippage')
        ]
        slippage_model = self._display_menu("Select Slippage Model", slippage_options)
        if slippage_model == 'back':
            return None
        params['slippage_model'] = slippage_model
        
        # Slippage value
        if slippage_model != 'none':
            while True:
                try:
                    slippage_value = float(input(f"\n{Fore.GREEN}Enter slippage value: {Style.RESET_ALL}").strip())
                    if slippage_value >= 0:
                        params['slippage_value'] = slippage_value
                        break
                    print(f"{Fore.RED}Value must be non-negative.{Style.RESET_ALL}")
                except ValueError:
                    print(f"{Fore.RED}Please enter a valid number.{Style.RESET_ALL}")
        
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
    
    def _get_boolean_input(self, prompt: str) -> bool:
        """Get boolean input from user."""
        while True:
            response = input(f"\n{Fore.GREEN}{prompt}{Style.RESET_ALL}").strip().lower()
            if response in ['y', 'yes', '1', 'true']:
                return True
            if response in ['n', 'no', '0', 'false']:
                return False
            print(f"{Fore.RED}Please enter 'y' or 'n'.{Style.RESET_ALL}")
    
    def _backtest_flow(self):
        """Handle the backtesting flow."""
        try:
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
            
            # 7. Configure Additional Parameters
            backtesting_params = self._configure_backtesting_parameters()
            if backtesting_params is None:
                return
            
            backward_testing_params = self._configure_backward_testing_parameters()
            if backward_testing_params is None:
                return
            
            statistical_params = self._configure_statistical_parameters()
            if statistical_params is None:
                return
            
            trade_engine_params = self._configure_trade_engine_parameters()
            if trade_engine_params is None:
                return
            
            risk_management_params = self._configure_risk_management_parameters()
            if risk_management_params is None:
                return
            
            performance_metrics_params = self._configure_performance_metrics_parameters()
            if performance_metrics_params is None:
                return
            
            data_validation_params = self._configure_data_validation_parameters()
            if data_validation_params is None:
                return
            
            # Add additional parameters
            tester_params.update({
                'ticker': instrument_key,
                'timeframe': f"{data_params['interval']}{data_params['unit']}"
            })
            
            # 8. Run Strategy
            print(f"\n{Fore.CYAN}Running strategy...{Style.RESET_ALL}")
            results = self.engine.run_strategy(
                data=data,
                strategy_name=strategy_name,
                strategy_params=strategy_params,
                tester_params=tester_params,
                **backtesting_params,
                **backward_testing_params,
                **statistical_params,
                **trade_engine_params,
                **risk_management_params,
                **performance_metrics_params,
                **data_validation_params
            )
            
            # 9. Display Results
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
            logger.error(f"Error in backtest flow: {str(e)}")
            print(f"\n{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
    
    def _optimization_flow(self):
        """Handle the optimization flow."""
        try:
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
            
            # 5. Configure Parameter Ranges
            strategy_info = self.engine.get_available_strategies()[strategy_name]
            param_ranges = {}
            
            print(f"\n{Fore.CYAN}=== Parameter Ranges for Optimization ==={Style.RESET_ALL}")
            for param_name, param_info in strategy_info['params'].items():
                print(f"\n{Fore.GREEN}{param_name} ({param_info['type']}):{Style.RESET_ALL}")
                print(f"Description: {param_info['description']}")
                
                if param_info['type'] in ['int', 'float']:
                    while True:
                        try:
                            min_val = float(input(f"Enter minimum value: ").strip())
                            max_val = float(input(f"Enter maximum value: ").strip())
                            step = float(input(f"Enter step size: ").strip())
                            
                            if min_val < max_val and step > 0:
                                param_ranges[param_name] = {
                                    'min': min_val,
                                    'max': max_val,
                                    'step': step
                                }
                                break
                            print(f"{Fore.RED}Invalid range or step size!{Style.RESET_ALL}")
                        except ValueError:
                            print(f"{Fore.RED}Please enter valid numbers.{Style.RESET_ALL}")
                else:
                    print(f"{Fore.YELLOW}Skipping non-numeric parameter.{Style.RESET_ALL}")
            
            # 6. Configure Optimizer Parameters
            optimizer_params = self._configure_optimizer_parameters()
            if optimizer_params is None:
                return
            
            # 7. Run Optimization
            print(f"\n{Fore.CYAN}Running optimization...{Style.RESET_ALL}")
            results = self.engine.optimize_strategy(
                data=data,
                strategy_name=strategy_name,
                param_ranges=param_ranges,
                optimization_params=optimizer_params
            )
            
            # 8. Display Results
            print(f"\n{Fore.GREEN}Optimization Results:{Style.RESET_ALL}")
            print("-" * 50)
            
            if 'best_params' in results:
                print("\nBest Parameters:")
                for param, value in results['best_params'].items():
                    print(f"{param}: {value}")
            
            if 'performance' in results:
                print("\nPerformance Metrics:")
                perf_df = pd.DataFrame(results['performance'])
                print(tabulate(perf_df, headers='keys', tablefmt='psql'))
            
            if 'recommendation' in results:
                print(f"\nRecommendation: {results['recommendation']}")
            
        except Exception as e:
            logger.error(f"Error in optimization flow: {str(e)}")
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