import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import logging

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclass
class BacktestMetadata:
    """Metadata for a backtest run"""
    timestamp: str
    ticker: str
    strategy_name: str
    start_date: str
    end_date: str
    initial_balance: float
    final_balance: float
    total_trades: int
    total_commission: float
    net_pnl: float
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    recommendation: str

class BacktestDataManager:
    """Manages the storage and retrieval of backtest data"""
    
    def __init__(self, base_dir: str = None):
        """
        Initialize the data manager
        
        Args:
            base_dir: Base directory for storing backtest data
        """
        if base_dir is None:
            self.base_dir = os.path.join('src', 'strategy_module', 'results', 'backtest')
        else:
            self.base_dir = base_dir
            
        # Create directory structure
        self._create_directory_structure()
        
        # Initialize data files if they don't exist
        self._initialize_data_files()
        
    def _create_directory_structure(self):
        """Create the directory structure for storing backtest data"""
        os.makedirs(self.base_dir, exist_ok=True)
            
    def _generate_timestamp(self) -> str:
        """Generate a unique timestamp for the backtest run"""
        return datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def _get_file_path(self, category: str) -> str:
        """Get the file path for a specific category"""
        return os.path.join(self.base_dir, f'{category}.json')
    
    def _initialize_data_files(self):
        """Initialize data files if they don't exist"""
        categories = ['trades', 'equity', 'metrics', 'market']
        for category in categories:
            file_path = self._get_file_path(category)
            if not os.path.exists(file_path):
                self._save_json(file_path, {})
    
    def _load_json(self, file_path: str) -> Dict:
        """Load data from a JSON file"""
        try:
            if not os.path.exists(file_path):
                return {}
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {str(e)}")
            raise
    
    def _save_json(self, file_path: str, data: Dict):
        """Save data to a JSON file"""
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4, cls=DateTimeEncoder)
        except Exception as e:
            logger.error(f"Error saving JSON file {file_path}: {str(e)}")
            raise
    
    def _get_run_key(self, strategy_name: str, timestamp: str) -> str:
        """Generate a unique key for a backtest run"""
        return f"{strategy_name}_{timestamp}"
    
    def save_backtest_results(self, results: Dict[str, Any], ticker: str, strategy_name: str) -> str:
        """
        Save backtest results in an organized structure
        
        Args:
            results: Dictionary containing backtest results
            ticker: Ticker symbol
            strategy_name: Name of the strategy
            
        Returns:
            str: Timestamp of the backtest run
        """
        try:
            timestamp = self._generate_timestamp()
            run_key = self._get_run_key(strategy_name, timestamp)
            
            # Load existing data
            trades_data = self._load_json(self._get_file_path('trades'))
            equity_data = self._load_json(self._get_file_path('equity'))
            metrics_data = self._load_json(self._get_file_path('metrics'))
            market_data = self._load_json(self._get_file_path('market'))
            
            # Update data with new run
            trades_data[run_key] = {
                'trades': results['trades'],
                'timestamp': timestamp,
                'ticker': ticker,
                'strategy_name': strategy_name
            }
            
            equity_data[run_key] = {
                'balance_history': results['balance_history'].tolist(),
                'equity_curve': results['equity_curve'].tolist(),
                'drawdown_curve': results['drawdown_curve'].tolist(),
                'timestamp': timestamp,
                'ticker': ticker,
                'strategy_name': strategy_name
            }
            
            metrics_data[run_key] = {
                'risk_metrics': results['risk_metrics'],
                'evaluation_metrics': results.get('evaluation_metrics', {}),
                'timestamp': timestamp,
                'ticker': ticker,
                'strategy_name': strategy_name
            }
            
            market_data[run_key] = {
                'timestamps': results['market_data']['timestamps'],
                'prices': results['market_data']['prices'],
                'signals': results['market_data']['signals'],
                'sl_levels': results['market_data']['sl_levels'],
                'tp_levels': results['market_data']['tp_levels'],
                'atr': results['market_data']['atr'],
                'returns': results['market_data']['returns'],
                'timestamp': timestamp,
                'ticker': ticker,
                'strategy_name': strategy_name
            }
            
            # Create metadata
            metadata = BacktestMetadata(
                timestamp=timestamp,
                ticker=ticker,
                strategy_name=strategy_name,
                start_date=results['market_data']['timestamps'][0],
                end_date=results['market_data']['timestamps'][-1],
                initial_balance=float(results['balance_history'][0]),
                final_balance=float(results['balance_history'][-1]),
                total_trades=len(results['trades']),
                total_commission=sum(trade['Total_Commission'] for trade in results['trades']),
                net_pnl=sum(trade['PnL'] for trade in results['trades']),
                win_rate=results['risk_metrics'].get('win_rate', 0.0),
                profit_factor=results['risk_metrics'].get('profit_factor', 0.0),
                sharpe_ratio=results['risk_metrics'].get('sharpe', 0.0),
                max_drawdown=results['risk_metrics'].get('max_drawdown', 0.0),
                recommendation=results.get('evaluation_metrics', {}).get('recommendation', '')
            )
            
            # Add metadata to metrics data
            metrics_data[run_key]['metadata'] = asdict(metadata)
            
            # Save all data files
            self._save_json(self._get_file_path('trades'), trades_data)
            self._save_json(self._get_file_path('equity'), equity_data)
            self._save_json(self._get_file_path('metrics'), metrics_data)
            self._save_json(self._get_file_path('market'), market_data)
            
            logger.info(f"Successfully saved backtest results with key: {run_key}")
            return run_key
            
        except Exception as e:
            logger.error(f"Error saving backtest results: {str(e)}")
            raise
    
    def load_backtest_results(self, run_key: str) -> Dict[str, Any]:
        """
        Load backtest results for a specific run
        
        Args:
            run_key: Unique key for the backtest run (strategy_name_timestamp)
            
        Returns:
            Dict containing all backtest results
        """
        try:
            # Load all data
            trades_data = self._load_json(self._get_file_path('trades'))
            equity_data = self._load_json(self._get_file_path('equity'))
            metrics_data = self._load_json(self._get_file_path('metrics'))
            market_data = self._load_json(self._get_file_path('market'))
            
            if run_key not in trades_data:
                raise KeyError(f"No backtest results found for key: {run_key}")
            
            # Combine all data
            results = {
                'metadata': metrics_data[run_key]['metadata'],
                'trades': trades_data[run_key],
                'equity': equity_data[run_key],
                'metrics': metrics_data[run_key],
                'market': market_data[run_key]
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error loading backtest results: {str(e)}")
            raise
    
    def get_strategy_runs(self, strategy_name: str) -> List[str]:
        """
        Get all run keys for a specific strategy
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            List of run keys
        """
        try:
            metrics_data = self._load_json(self._get_file_path('metrics'))
            return [key for key in metrics_data.keys() if key.startswith(f"{strategy_name}_")]
        except Exception as e:
            logger.error(f"Error getting strategy runs: {str(e)}")
            raise
    
    def get_latest_run(self, strategy_name: str) -> Optional[str]:
        """
        Get the latest run key for a specific strategy
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Latest run key or None if no runs exist
        """
        try:
            runs = self.get_strategy_runs(strategy_name)
            if not runs:
                return None
            return max(runs, key=lambda x: x.split('_')[-1])
        except Exception as e:
            logger.error(f"Error getting latest run: {str(e)}")
            raise

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling datetime and numpy objects"""
    def default(self, obj):
        if isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)):
            return obj.isoformat()
        elif isinstance(obj, pd.Timedelta):
            return str(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj) 