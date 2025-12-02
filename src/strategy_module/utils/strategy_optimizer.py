import logging
from typing import Dict, List, Optional, Tuple, Type
import pandas as pd
import numpy as np
from datetime import datetime
import os
import json
from dotenv import load_dotenv
from pathlib import Path

from src.broker_module.upstox.data.CandleData import UpstoxHistoricalData
from src.broker_module.upstox.utils.InstrumentKeyFinder import InstrumentKeyFinder
from itertools import product
from scipy.optimize import differential_evolution
from src.strategy_module.utils.backward_testing import BacktestingEngine
from src.strategy_module.utils.strategy_evaluation_metrics import StrategyEvaluationMetrics
from src.strategy_module.strategies.rsi_strategy import RSIStrategy
from src.strategy_module.strategies.moving_average_strategy import MovingAverageStrategy
from src.strategy_module.strategies.time_based_strategy import TimeBasedStrategy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StrategyOptimizer:
    def __init__(self, 
                 ticker: str,
                 strategy_class: Type,
                 parameter_bounds: List[tuple],
                 data: pd.DataFrame,  # Made mandatory
                 criterion: str = "Balance_Max",
                 timeframe: str = "5minute",
                 trade_engine_type: str = "Signals",
                 optimizing_period: int = 3,
                 balance: float = 1000,
                 leverage: float = 500,
                 volume: int = 1,
                 commission: float = 7,
                 optimization_method: str = "grid_search",
                 max_combinations: int = 1000):
        """
        Initialize strategy optimizer with enhanced optimization capabilities.
        
        Args:
            ticker: Trading symbol
            strategy_class: Strategy class to optimize
            parameter_bounds: List of tuples containing parameter bounds
            data: Input DataFrame containing historical price data
            criterion: Optimization criterion
            timeframe: Trading timeframe
            trade_engine_type: Type of trade engine
            optimizing_period: Optimization period in months
            balance: Initial account balance
            leverage: Account leverage
            volume: Trading volume
            commission: Commission per trade
            optimization_method: Method to use for optimization ('grid_search' or 'differential_evolution')
            max_combinations: Maximum number of parameter combinations to test
        """
        self._ticker = ticker
        self._strategy_class = strategy_class
        self._parameter_bounds = parameter_bounds if parameter_bounds else self._get_default_parameter_bounds()
        self._criterion = criterion
        self._timeframe = timeframe
        self._trade_engine_type = trade_engine_type
        self._optimizing_period = optimizing_period
        self._balance = balance
        self._leverage = leverage
        self._volume = volume
        self._commission = commission
        self._optimization_method = optimization_method
        self._max_combinations = max_combinations
        
        # Initialize results container
        self.optimized_stats = pd.DataFrame(columns=[
            'Ticker', 'Parameters', 'Win_Rate_Max', 'Balance_Max', 
            'Drawdown_Min', 'Profit_Factor_Max', 'Recovery_Factor_Max',
            'Sharpe_Ratio_Max', 'Sortino_Ratio_Max', 'Calmar_ratio_Max',
            'Expected_Payoff_Max'
        ])
        
        # Make a copy of input data to prevent modifications
        self.data = data.copy()
        
        # Validate data
        self._validate_data()
        
        # Generate parameter combinations
        self._generate_combinations()
        
        # Run optimization
        self.optimize_strategy()

    def _get_default_parameter_bounds(self) -> List[tuple]:
        """Get default parameter bounds based on strategy class."""
        try:
            # Get strategy parameters from class
            param_names = self._strategy_class.__init__.__code__.co_varnames[1:]
            param_defaults = self._strategy_class.__init__.__defaults__ or ()
            
            # Create default bounds
            bounds = []
            for i, param_name in enumerate(param_names):
                if i < len(param_defaults):
                    default_value = param_defaults[i]
                    if isinstance(default_value, (int, float)):
                        # Create a range around the default value
                        lower = max(1, int(default_value * 0.5))
                        upper = int(default_value * 1.5)
                        bounds.append((lower, upper))
            
            if not bounds:
                # If no numeric parameters found, create some default bounds
                bounds = [(1, 20), (1, 50), (1, 100)]
            
            logger.info(f"Generated default parameter bounds: {bounds}")
            return bounds
            
        except Exception as e:
            logger.error(f"Error generating default parameter bounds: {str(e)}")
            # Return some safe default bounds
            return [(1, 20), (1, 50), (1, 100)]

    def _validate_data(self) -> None:
        """Validate input data structure and content."""
        if self.data is None or self.data.empty:
            raise ValueError("Input data cannot be empty")
            
        required_columns = ['time', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Data missing required columns: {missing_columns}")
            
        # Check for missing values
        if self.data[required_columns].isnull().any().any():
            logger.warning("Data contains missing values. Filling with forward fill method.")
            self.data[required_columns] = self.data[required_columns].ffill()
            
        # Ensure time column is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.data['time']):
            self.data['time'] = pd.to_datetime(self.data['time'], errors='coerce', infer_datetime_format=True)
            
        # Sort data by time
        self.data = self.data.sort_values('time')
        
        logger.info(f"Validated input data with {len(self.data)} rows")

    def _generate_combinations(self) -> None:
        """Generate parameter combinations using selected optimization method."""
        if self._optimization_method == "grid_search":
            self._generate_grid_combinations()
        elif self._optimization_method == "differential_evolution":
            self._prepare_differential_evolution()
        else:
            raise ValueError(f"Unsupported optimization method: {self._optimization_method}")

    def _generate_grid_combinations(self) -> List[np.ndarray]:
        """Generate parameter combinations using grid search with sampling."""
        try:
            # Calculate total combinations
            total_combinations = np.prod([end - start + 1 for start, end in self._parameter_bounds])
            
            if total_combinations > self._max_combinations:
                logger.warning(f"Total combinations ({total_combinations}) exceeds max_combinations ({self._max_combinations}). Using sampling.")
                
                # Generate sampled combinations
                param_ranges = []
                for start, end in self._parameter_bounds:
                    # Calculate step size to get approximately max_combinations points
                    step = max(1, int((end - start + 1) / (self._max_combinations ** (1/len(self._parameter_bounds)))))
                    param_range = np.arange(start, end + 1, step)
                    if len(param_range) == 0:  # If step size is too large
                        param_range = np.array([start, end])
                    param_ranges.append(param_range)
                
                # Generate combinations using meshgrid
                mesh = np.meshgrid(*param_ranges)
                combinations = np.column_stack([m.ravel() for m in mesh])
                
                # If still too many combinations, randomly sample
                if len(combinations) > self._max_combinations:
                    indices = np.random.choice(len(combinations), self._max_combinations, replace=False)
                    combinations = combinations[indices]
                
                # Convert to integers
                combinations = combinations.astype(int)
                
                # Validate combinations
                valid_combinations = []
                for combo in combinations:
                    if all(start <= val <= end for (start, end), val in zip(self._parameter_bounds, combo)):
                        valid_combinations.append(combo)
                
                if not valid_combinations:
                    logger.warning("No valid combinations generated, using default range")
                    # Generate some safe default combinations
                    valid_combinations = np.array([[start for start, _ in self._parameter_bounds]])
                
                logger.info(f"Generated {len(valid_combinations)} valid parameter combinations")
                return valid_combinations
                
            else:
                # Use full grid search
                param_ranges = [np.arange(start, end + 1) for start, end in self._parameter_bounds]
                mesh = np.meshgrid(*param_ranges)
                combinations = np.column_stack([m.ravel() for m in mesh])
                combinations = combinations.astype(int)
                
                logger.info(f"Generated {len(combinations)} parameter combinations")
                return combinations
            
        except Exception as e:
            logger.error(f"Error generating parameter combinations: {str(e)}")
            # Generate some safe default combinations
            default_combinations = np.array([[start for start, _ in self._parameter_bounds]])
            logger.info("Using default parameter combinations")
            return default_combinations

    def _prepare_differential_evolution(self) -> None:
        """Prepare for differential evolution optimization."""
        # Convert bounds to list of tuples for scipy
        self._bounds = [(start, end) for start, end in self._parameter_bounds]
        
        # Initialize population size based on parameter space
        self._popsize = min(20 * len(self._bounds), self._max_combinations)
        
        # Initialize mutation and crossover parameters
        self._mutation = (0.5, 1.0)  # (F_lower, F_upper) for mutation
        self._recombination = 0.7    # CR for crossover
        self._strategy = 'best1bin'  # Mutation strategy
        
        # Initialize convergence tracking
        self._best_fitness_history = []
        self._generation_count = 0
        self._max_generations = 50
        self._tolerance = 1e-4
        self._patience = 10  # Number of generations to wait for improvement
        
        logger.info(f"Prepared DE optimization with {self._popsize} population size")

    def _initialize_population(self) -> np.ndarray:
        """Initialize population for differential evolution."""
        population = []
        for _ in range(self._popsize):
            # Generate random parameters within bounds
            params = []
            for (lower, upper) in self._bounds:
                param = np.random.randint(lower, upper + 1)
                params.append(param)
            population.append(params)
        return np.array(population)

    def _mutate(self, population: np.ndarray, best_idx: int) -> np.ndarray:
        """Apply mutation to create new candidate solutions."""
        mutant_population = np.zeros_like(population)
        
        for i in range(self._popsize):
            if self._strategy == 'best1bin':
                # Use best solution and two random solutions
                idxs = [idx for idx in range(self._popsize) if idx != i]
                a, b = population[np.random.choice(idxs, 2, replace=False)]
                best = population[best_idx]
                
                # Calculate mutation
                F = np.random.uniform(self._mutation[0], self._mutation[1])
                mutant = best + F * (a - b)
                
                # Ensure integer values and bounds
                mutant = np.clip(mutant, [b[0] for b in self._bounds], [b[1] for b in self._bounds])
                mutant = np.round(mutant).astype(int)
                
                mutant_population[i] = mutant
        
        return mutant_population

    def _crossover(self, population: np.ndarray, mutant_population: np.ndarray) -> np.ndarray:
        """Apply crossover to combine parent and mutant solutions."""
        trial_population = np.zeros_like(population)
        
        for i in range(self._popsize):
            # Crossover mask
            mask = np.random.random(len(self._bounds)) < self._recombination
            
            # Ensure at least one parameter is changed
            if not np.any(mask):
                mask[np.random.randint(0, len(self._bounds))] = True
            
            # Create trial solution
            trial = np.where(mask, mutant_population[i], population[i])
            trial_population[i] = trial
        
        return trial_population

    def _select(self, population: np.ndarray, trial_population: np.ndarray) -> np.ndarray:
        """Select better solutions between parent and trial populations."""
        new_population = np.zeros_like(population)
        
        for i in range(self._popsize):
            # Evaluate both solutions
            parent_fitness = self._objective_function(population[i])
            trial_fitness = self._objective_function(trial_population[i])
            
            # Select better solution
            if trial_fitness > parent_fitness:
                new_population[i] = trial_population[i]
            else:
                new_population[i] = population[i]
        
        return new_population

    def _check_convergence(self, best_fitness: float) -> bool:
        """Check if optimization has converged."""
        self._best_fitness_history.append(best_fitness)
        
        # Check maximum generations
        if self._generation_count >= self._max_generations:
            logger.info(f"Optimization reached maximum generations ({self._max_generations})")
            return True
        
        # Check if we have enough history to check convergence
        if len(self._best_fitness_history) > self._patience:
            # Calculate improvement in last 'patience' generations
            recent_history = self._best_fitness_history[-self._patience:]
            recent_improvement = max(recent_history) - min(recent_history)
            
            # Log convergence status
            logger.debug(f"Recent improvement: {recent_improvement:.6f}, Tolerance: {self._tolerance}")
            
            if recent_improvement < self._tolerance:
                logger.info(f"Optimization converged - improvement below tolerance ({self._tolerance})")
                return True
        
        return False

    def _instantiate_strategy(self, params: tuple) -> bool:
        """Instantiate strategy with given parameters."""
        try:
            param_names = self._strategy_class.__init__.__code__.co_varnames[1:]
            param_dict = dict(zip(param_names, params))
            self.strategy_instance = self._strategy_class(**param_dict)
            return True
        except Exception as e:
            logger.error(f"Error instantiating strategy: {str(e)}")
            return False

    def _evaluate_strategy(self, params: np.ndarray) -> Optional[Dict]:
        """Evaluate strategy with given parameters."""
        try:
            # Convert parameters to integers
            params_tuple = tuple(int(x) for x in params)
            
            if not self._instantiate_strategy(params_tuple):
                logger.warning("Failed to instantiate strategy")
                return None
            
            # Generate signals
            signals = self.strategy_instance.generate_signals(self.data)
            
            if signals is None or signals.empty:
                logger.warning("No signals generated for this parameter combination")
                return None
            
            # Validate signals
            if 'Signal' not in signals.columns:
                logger.warning("Missing 'Signal' column in generated signals")
                return None
            
            # Prepare input data
            required_columns = ['time', 'close', 'Signal']
            if not all(col in signals.columns for col in required_columns):
                logger.warning(f"Missing required columns: {required_columns}")
                return None
            
            # Initialize trade engine
            trade_engine = BacktestingEngine()
            
            # Execute trades
            if self._trade_engine_type == "Signals":
                input_data = signals[required_columns].copy()
                input_data.columns = ['time', 'Price', 'Signal']
                results = trade_engine.signal_based_trade_executor(
                    input_data, 
                    self._balance, 
                    self._leverage, 
                    self._volume, 
                    self._commission
                )
            else:  # SLTP
                sltp_columns = required_columns + ['SL', 'TP']
                if not all(col in signals.columns for col in sltp_columns):
                    logger.warning("Missing SL/TP columns")
                    return None
                input_data = signals[sltp_columns].copy()
                input_data.columns = ['time', 'Price', 'Signal', 'SL', 'TP']
                results = trade_engine.sltp_based_trade_executor(
                    input_data, 
                    self._balance, 
                    self._leverage, 
                    self._volume, 
                    self._commission
                )
            
            if not results or 'trades' not in results:
                logger.warning("No trade results generated")
                return None
                
            trades = pd.DataFrame(results['trades'])
            
            if trades.empty:
                logger.warning("No trades executed for this parameter combination")
                return None
            
            # Calculate metrics
            strategy_metrics = StrategyEvaluationMetrics(trades, self._balance, self._ticker)
            metrics = strategy_metrics.metrics
            stats = strategy_metrics.stats
            
            if metrics is None or stats is None:
                logger.warning("Failed to calculate metrics")
                return None
            
            # Create results dictionary with safe access to metrics and stats
            result_dict = {
                'Ticker': self._ticker,
                'Parameters': params_tuple,
                'Win_Rate_Max': 0.0,
                'Balance_Max': 0.0,
                'Drawdown_Min': 0.0,
                'Profit_Factor_Max': 0.0,
                'Recovery_Factor_Max': 0.0,
                'Sharpe_Ratio_Max': 0.0,
                'Sortino_Ratio_Max': 0.0,
                'Calmar_ratio_Max': 0.0,
                'Expected_Payoff_Max': 0.0
            }
            
            # Safely access metrics
            try:
                balance_metric = metrics[metrics['Metric'] == 'Final_Balance']
                if not balance_metric.empty:
                    result_dict['Balance_Max'] = float(balance_metric['Value'].iloc[0])
                
                drawdown_metric = metrics[metrics['Metric'] == 'Balance_Absolute_Drawdown']
                if not drawdown_metric.empty:
                    result_dict['Drawdown_Min'] = float(drawdown_metric['Value'].iloc[0])
            except Exception as e:
                logger.warning(f"Error accessing metrics: {str(e)}")
            
            # Safely access statistics
            try:
                if not stats.empty:
                    for stat_name, result_key in [
                        ('Profit_Factor', 'Profit_Factor_Max'),
                        ('Recovery_Factor', 'Recovery_Factor_Max'),
                        ('Sharpe_Ratio', 'Sharpe_Ratio_Max'),
                        ('Sortino_ratio', 'Sortino_Ratio_Max'),
                        ('calmar_ratio', 'Calmar_ratio_Max'),
                        ('Expected_Payoff', 'Expected_Payoff_Max'),
                        ('Win_Rate', 'Win_Rate_Max')
                    ]:
                        stat_value = stats[stats['Statistic'] == stat_name]
                        if not stat_value.empty:
                            result_dict[result_key] = float(stat_value['Value'].iloc[0])
            except Exception as e:
                logger.warning(f"Error accessing statistics: {str(e)}")
            
            return result_dict
            
        except Exception as e:
            logger.error(f"Error evaluating strategy: {str(e)}")
            return None

    def _objective_function(self, params: np.ndarray) -> float:
        """Objective function for differential evolution."""
        results = self._evaluate_strategy(params)
        if results is None:
            return -np.inf
        return results[self._criterion]

    def optimize_strategy(self) -> None:
        """Run strategy optimization using selected method."""
        logger.info("Starting strategy optimization...")
        
        if self._optimization_method == "grid_search":
            self._optimize_grid_search()
        else:  # differential_evolution
            self._optimize_differential_evolution()

    def _optimize_grid_search(self) -> None:
        """Run grid search optimization."""
        try:
            # Generate parameter combinations
            param_combinations = self._generate_grid_combinations()
            
            if param_combinations is None or len(param_combinations) == 0:
                logger.warning("No valid parameter combinations generated")
                self.optimized_stats = pd.DataFrame()
                return
            
            # Initialize results list
            results = []
            
            # Evaluate each combination
            total_combinations = len(param_combinations)
            for i, params in enumerate(param_combinations, 1):
                logger.info(f"Evaluating combination {i}/{total_combinations}")
                result = self._evaluate_strategy(params)
                if result is not None:
                    results.append(result)
            
            if not results:
                logger.warning("No valid results from strategy evaluation")
                self.optimized_stats = pd.DataFrame()
                return
            
            # Convert results to DataFrame
            self.optimized_stats = pd.DataFrame(results)
            
            # Ensure all required columns exist
            required_columns = [
                'Ticker', 'Parameters', 'Win_Rate_Max', 'Balance_Max', 
                'Drawdown_Min', 'Profit_Factor_Max', 'Recovery_Factor_Max',
                'Sharpe_Ratio_Max', 'Sortino_Ratio_Max', 'Calmar_ratio_Max',
                'Expected_Payoff_Max'
            ]
            
            for col in required_columns:
                if col not in self.optimized_stats.columns:
                    self.optimized_stats[col] = 0.0
            
            # Sort by optimization criterion
            if self._criterion in self.optimized_stats.columns:
                self.optimized_stats = self.optimized_stats.sort_values(
                    by=self._criterion,
                    ascending=False
                )
            
            logger.info(f"Grid search completed with {len(results)} valid results")
            
        except Exception as e:
            logger.error(f"Error in grid search optimization: {str(e)}")
            self.optimized_stats = pd.DataFrame()

    def _optimize_differential_evolution(self) -> None:
        """Run differential evolution optimization with improved implementation."""
        try:
            logger.info("Starting differential evolution optimization...")
            
            # Initialize population
            population = self._initialize_population()
            best_fitness = -np.inf
            best_solution = None
            no_improvement_count = 0
            last_best_fitness = -np.inf
            
            # Main optimization loop
            while True:
                self._generation_count += 1
                logger.info(f"Generation {self._generation_count}")
                
                # Find best solution in current population
                fitness_values = []
                for ind in population:
                    try:
                        fitness = self._objective_function(ind)
                        fitness_values.append(fitness)
                    except Exception as e:
                        logger.warning(f"Error evaluating individual: {str(e)}")
                        fitness_values.append(-np.inf)
                
                best_idx = np.argmax(fitness_values)
                current_best_fitness = fitness_values[best_idx]
                
                # Check for improvement
                if current_best_fitness > best_fitness:
                    best_fitness = current_best_fitness
                    best_solution = population[best_idx].copy()
                    no_improvement_count = 0
                    logger.info(f"New best fitness: {best_fitness:.4f}")
                else:
                    no_improvement_count += 1
                
                # Check for stagnation
                if no_improvement_count >= 10:
                    logger.info("No improvement for 10 generations, stopping optimization")
                    break
                
                # Check convergence
                if self._check_convergence(best_fitness):
                    break
                
                # Apply mutation
                try:
                    mutant_population = self._mutate(population, best_idx)
                except Exception as e:
                    logger.error(f"Error in mutation: {str(e)}")
                    break
                
                # Apply crossover
                try:
                    trial_population = self._crossover(population, mutant_population)
                except Exception as e:
                    logger.error(f"Error in crossover: {str(e)}")
                    break
                
                # Select new population
                try:
                    population = self._select(population, trial_population)
                except Exception as e:
                    logger.error(f"Error in selection: {str(e)}")
                    break
            
            # Store results
            if best_solution is not None and best_fitness > -np.inf:
                best_result = self._evaluate_strategy(best_solution)
                if best_result is not None:
                    self.optimized_stats = pd.DataFrame([best_result])
                    logger.info("Differential evolution optimization completed")
                    logger.info(f"Best parameters: {best_solution}")
                    logger.info(f"Best fitness: {best_fitness:.4f}")
                    logger.info(f"Total generations: {self._generation_count}")
                else:
                    logger.warning("Failed to evaluate best parameters")
                    self._fallback_to_grid_search()
            else:
                logger.warning("No valid solution found")
                self._fallback_to_grid_search()
                
        except Exception as e:
            logger.error(f"Error in differential evolution: {str(e)}")
            self._fallback_to_grid_search()

    def _fallback_to_grid_search(self):
        """Fallback to grid search when differential evolution fails."""
        logger.warning("Falling back to grid search")
        self._optimization_method = "grid_search"
        self._generate_combinations()
        self._optimize_grid_search()

    def get_optimization_results(self) -> pd.DataFrame:
        """Get optimization results."""
        try:
            # Initialize empty DataFrame with required columns
            required_columns = [
                'Ticker', 'Parameters', 'Win_Rate_Max', 'Balance_Max', 
                'Drawdown_Min', 'Profit_Factor_Max', 'Recovery_Factor_Max',
                'Sharpe_Ratio_Max', 'Sortino_Ratio_Max', 'Calmar_ratio_Max',
                'Expected_Payoff_Max'
            ]
            
            # Check if we have any results
            if self.optimized_stats is None or self.optimized_stats.empty:
                logger.warning("No optimization results available")
                return pd.DataFrame(columns=required_columns)
            
            # Create a copy of the results
            results_df = self.optimized_stats.copy()
            
            # Ensure all required columns exist
            for col in required_columns:
                if col not in results_df.columns:
                    results_df[col] = 0.0
            
            # Convert Parameters to tuple if it's not already
            if 'Parameters' in results_df.columns:
                results_df['Parameters'] = results_df['Parameters'].apply(
                    lambda x: tuple(x) if isinstance(x, (list, np.ndarray)) else x
                )
            
            # Fill any NaN values with 0
            results_df = results_df.fillna(0.0)
            
            # Sort by criterion if available
            if self._criterion in results_df.columns:
                results_df = results_df.sort_values(
                    by=self._criterion,
                    ascending=False
                )
            
            # Validate the DataFrame
            if results_df.empty:
                logger.warning("No valid results after processing")
                return pd.DataFrame(columns=required_columns)
            
            logger.info(f"Returning {len(results_df)} optimization results")
            return results_df
            
        except Exception as e:
            logger.error(f"Error getting optimization results: {str(e)}")
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=required_columns)

    def get_best_parameters(self) -> Optional[Tuple]:
        """Get best parameters based on optimization criterion."""
        try:
            if self.optimized_stats is None or self.optimized_stats.empty:
                logger.warning("No optimization results available")
                return None
                
            # Get the best result
            best_result = self.optimized_stats.iloc[0]
            
            # Convert parameters to tuple if needed
            params = best_result['Parameters']
            if isinstance(params, (list, np.ndarray)):
                params = tuple(params)
            
            logger.info(f"Best parameters: {params}")
            return params
            
        except Exception as e:
            logger.error(f"Error getting best parameters: {str(e)}")
            return None

    def store_optimization_results(self, output_dir: str = None) -> None:
        """Store optimization results in separate JSON files.
        
        Args:
            output_dir (str, optional): Directory to store the JSON files. 
                                      If None, uses a default directory.
        """
        logger.info("Storing optimization results...")
        
        try:
            # Create output directory if not provided
            if output_dir is None:
                output_dir = os.path.join(
                    os.path.dirname(__file__),
                    '..',
                    'results',
                    'optimization'
                )
            
            # Create directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate unique test ID with strategy name and symbol
            strategy_name = self._strategy_class.__name__ if hasattr(self._strategy_class, '__name__') else str(self._strategy_class)
            symbol = self._ticker.split('|')[0] if '|' in self._ticker else self._ticker
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            test_id = f"{strategy_name}_{symbol}_{timestamp}"
            
            # Get optimization results
            results_df = self.get_optimization_results()
            
            # Function to serialize data
            def serialize_data(data):
                """Serialize data to JSON-compatible format."""
                if isinstance(data, (datetime, pd.Timestamp)):
                    return data.isoformat()
                if isinstance(data, pd.DataFrame):
                    return data.to_dict(orient='records')
                if isinstance(data, pd.Series):
                    return data.to_dict()
                if isinstance(data, np.ndarray):
                    return data.tolist()
                if isinstance(data, np.integer):
                    return int(data)
                if isinstance(data, np.floating):
                    return float(data)
                if isinstance(data, dict):
                    return {k: serialize_data(v) for k, v in data.items()}
                if isinstance(data, list):
                    return [serialize_data(item) for item in data]
                return data
            
            # Function to store data with test_id as primary key
            def store_data(file_path: str, data: Dict, test_id: str) -> None:
                try:
                    # Serialize data
                    serialized_data = serialize_data(data)
                    
                    # Create data structure with test_id as primary key
                    data_entry = {
                        test_id: {
                            'data': serialized_data,
                            'metadata': {
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'ticker': self._ticker,
                                'timeframe': self._timeframe,
                                'strategy_name': strategy_name
                            }
                        }
                    }
                    
                    # Read existing data
                    if os.path.exists(file_path):
                        with open(file_path, 'r') as f:
                            try:
                                existing_data = json.load(f)
                            except json.JSONDecodeError:
                                existing_data = {}
                    else:
                        existing_data = {}
                    
                    # Update existing data with new entry
                    existing_data.update(data_entry)
                    
                    # Write back to file
                    with open(file_path, 'w') as f:
                        json.dump(existing_data, f, indent=4)
                        
                    logger.info(f"Successfully stored data in {file_path}")
                    
                except Exception as e:
                    logger.error(f"Error storing data in {file_path}: {str(e)}")
                    # If error occurs, write new data as single entry
                    try:
                        with open(file_path, 'w') as f:
                            json.dump(data_entry, f, indent=4)
                        logger.info(f"Created new file {file_path} with single entry")
                    except Exception as write_error:
                        logger.error(f"Error writing to {file_path}: {str(write_error)}")
            
            # Store optimization results
            results_file = os.path.join(output_dir, 'results.json')
            store_data(results_file, results_df.to_dict('records'), test_id)
            
            # Store best parameters
            best_params = self.get_best_parameters()
            if best_params is not None:
                best_params_file = os.path.join(output_dir, 'best_parameters.json')
                store_data(best_params_file, {
                    'parameters': best_params,
                    'criterion': self._criterion,
                    'best_value': float(results_df[self._criterion].iloc[0]) if not results_df.empty else None
                }, test_id)
            
            # Store configuration
            config_file = os.path.join(output_dir, 'config.json')
            config = {
                'data_loading_params': {
                    'instrument_key': self._ticker,
                    'unit': self._timeframe.replace(str(self._timeframe.split('minute')[0]), '').strip() if 'minute' in self._timeframe else 'days',
                    'interval': int(self._timeframe.split('minute')[0]) if 'minute' in self._timeframe else 1,
                    'data_type': 'historical',
                    'days': self._optimizing_period * 30  # Convert months to days
                },
                'strategy_params': {
                    'parameter_bounds': self._parameter_bounds
                },
                'optimization_params': {
                    'criterion': self._criterion,
                    'optimization_method': self._optimization_method,
                    'max_combinations': self._max_combinations
                },
                'backtesting_params': {
                    'trade_engine_type': self._trade_engine_type,
                    'investing_amount': self._balance,
                    'account_leverage': self._leverage,
                    'volume': self._volume,
                    'commission': self._commission
                },
                'data_validation_params': {
                    'required_columns': ['time', 'open', 'high', 'low', 'close', 'volume'],
                    'data_validation': True
                }
            }
            store_data(config_file, config, test_id)
            
            # Store test index
            index_file = os.path.join(output_dir, 'test_index.json')
            index_entry = {
                'summary': {
                    'total_combinations_tested': len(results_df) if not results_df.empty else 0,
                    'best_criterion_value': float(results_df[self._criterion].iloc[0]) if not results_df.empty else None,
                    'generations_completed': self._generation_count if hasattr(self, '_generation_count') else None
                }
            }
            store_data(index_file, index_entry, test_id)
            
            logger.info(f"All optimization results stored with test_id: {test_id}")
            
        except Exception as e:
            logger.error(f"Error storing optimization results: {str(e)}")
            raise 

    @classmethod
    def load_config(cls, test_id: str, output_dir: str = None) -> Optional[Dict]:
        """Load a saved configuration by test ID.
        
        Args:
            test_id (str): Test ID to load
            output_dir (str, optional): Directory containing the configuration files.
                                      If None, uses the default directory.
        
        Returns:
            Optional[Dict]: Loaded configuration if found, None otherwise
        """
        logger.info(f"Loading configuration for test_id: {test_id}")
        
        try:
            # Create output directory if not provided
            if output_dir is None:
                output_dir = os.path.join(
                    os.path.dirname(__file__),
                    '..',
                    'results',
                    'optimization'
                )
            
            # Load config file
            config_file = os.path.join(output_dir, 'config.json')
            if not os.path.exists(config_file):
                logger.warning(f"Config file not found: {config_file}")
                return None
            
            with open(config_file, 'r') as f:
                configs = json.load(f)
                
                if test_id not in configs:
                    logger.warning(f"Test ID {test_id} not found in config file")
                    return None
                
                config = configs[test_id]
                
                # Extract data and metadata
                data = config.get('data', {})
                metadata = config.get('metadata', {})
                
                # Validate required fields
                required_fields = ['data_loading_params', 'strategy_params', 'optimization_params']
                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    logger.warning(f"Missing required fields in config: {missing_fields}")
                    return None
                
                # Create a complete configuration dictionary
                complete_config = {
                    'data': data,
                    'metadata': metadata
                }
                
                logger.info(f"Successfully loaded configuration for test_id: {test_id}")
                return complete_config
                
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return None

    @classmethod
    def load_saved_configs(cls, output_dir: str = None) -> List[Dict]:
        """Load all saved configurations.
        
        Args:
            output_dir (str, optional): Directory containing the configuration files.
                                      If None, uses the default directory.
        
        Returns:
            List[Dict]: List of saved configurations
        """
        logger.info("Loading saved configurations...")
        
        try:
            # Create output directory if not provided
            if output_dir is None:
                output_dir = os.path.join(
                    os.path.dirname(__file__),
                    '..',
                    'results',
                    'optimization'
                )
            
            # Load config file
            config_file = os.path.join(output_dir, 'config.json')
            if not os.path.exists(config_file):
                logger.warning(f"Config file not found: {config_file}")
                return []
            
            with open(config_file, 'r') as f:
                configs = json.load(f)
                
                # Convert dict of configs to list
                config_list = []
                for test_id, config in configs.items():
                    # Create a flattened configuration with test_id
                    flat_config = {
                        'test_id': test_id,
                        **config
                    }
                    config_list.append(flat_config)
                
                # Sort configs by timestamp
                config_list.sort(
                    key=lambda x: x.get('metadata', {}).get('timestamp', ''),
                    reverse=True
                )
                
                logger.info(f"Loaded {len(config_list)} configurations")
                return config_list
                
        except Exception as e:
            logger.error(f"Error loading saved configurations: {str(e)}")
            return [] 