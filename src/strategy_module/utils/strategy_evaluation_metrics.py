import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class StrategyMetrics:
    """Data class to store strategy evaluation metrics"""
    ticker: str
    initial_deposit: float
    final_balance: float
    total_net_profit: float
    gross_profit: float
    gross_loss: float
    balance_absolute_drawdown: float
    balance_maximal_drawdown_pct: float
    balance_relative_drawdown_pct: float
    commission_paid: float
    min_position_holding_time: pd.Timedelta
    max_position_holding_time: pd.Timedelta
    avg_position_holding_time: pd.Timedelta
    total_trades: int
    total_deals: int
    long_trades_won: float
    short_trades_won: float
    largest_profit_trade: float
    largest_loss_trade: float
    avg_profit_per_trade: float
    avg_loss_per_trade: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    avg_return_per_trade: float
    std_dev_returns: float

class StrategyEvaluationMetrics:
    def __init__(self, data: pd.DataFrame, balance: float, ticker: str):
        """
        Initialize strategy evaluation metrics calculator.
        
        Args:
            data: DataFrame containing trade data
            balance: Initial account balance
            ticker: Trading symbol
        """
        self.Trades = data
        self.balance = balance
        self.Ticker = ticker
        self.EPS = 1e-8  # Small threshold for numerical stability
        
        # Initialize metrics containers
        self.metrics = pd.DataFrame()
        self.stats = pd.DataFrame()
        self.Final_Recommendation = None
        
        # Calculate metrics
        self._calculate_consecutive_stats()
        self._calculate_evaluation_metrics()
        self._calculate_strategy_statistics()
        self._interpret_trading_metrics()

    def _calculate_consecutive_stats(self) -> None:
        """Calculate consecutive wins and losses using vectorized operations"""
        pnls = self.Trades['PnL'].values
        # Create masks for wins and losses
        wins = pnls > 0
        losses = pnls < 0
        
        # Calculate consecutive wins and losses
        win_streaks = np.zeros_like(pnls, dtype=int)
        loss_streaks = np.zeros_like(pnls, dtype=int)
        
        # Vectorized calculation of streaks
        win_streaks[wins] = 1
        loss_streaks[losses] = 1
        
        # Calculate cumulative sums
        win_streaks = np.cumsum(win_streaks)
        loss_streaks = np.cumsum(loss_streaks)
        
        # Reset streaks when opposite occurs
        win_streaks[losses] = 0
        loss_streaks[wins] = 0
        
        self._max_consecutive_wins = np.max(win_streaks)
        self._max_consecutive_losses = np.max(loss_streaks)

    def _calculate_evaluation_metrics(self) -> None:
        """Calculate evaluation metrics using vectorized operations"""
        try:
            # Convert to numpy arrays for faster calculations
            pnls = self.Trades['PnL'].values
            account_balances = self.Trades['Balance'].values
            
            # Calculate basic metrics
            total_trades = len(self.Trades)
            winning_trades = np.sum(pnls > 0)
            losing_trades = np.sum(pnls < 0)
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            # Calculate PnL metrics
            total_pnl = np.sum(pnls)
            average_pnl = np.mean(pnls) if total_trades > 0 else 0
            max_pnl = np.max(pnls) if total_trades > 0 else 0
            min_pnl = np.min(pnls) if total_trades > 0 else 0
            
            # Calculate drawdown metrics
            cumulative_returns = np.cumsum(pnls)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = running_max - cumulative_returns
            max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
            
            # Calculate risk metrics
            daily_returns = np.diff(account_balances) / account_balances[:-1]
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) if len(daily_returns) > 0 and np.std(daily_returns) != 0 else 0
            sortino_ratio = np.mean(daily_returns) / np.std(daily_returns[daily_returns < 0]) if len(daily_returns) > 0 and np.std(daily_returns[daily_returns < 0]) != 0 else 0
            
            # Calculate profit factor
            gross_profit = np.sum(pnls[pnls > 0])
            gross_loss = abs(np.sum(pnls[pnls < 0]))
            profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
            
            # Calculate final balance
            final_balance = account_balances[-1] if len(account_balances) > 0 else self.balance
            
            # Store metrics
            self.metrics = pd.DataFrame({
                'Metric': [
                    'Total_Trades', 'Winning_Trades', 'Losing_Trades', 'Win_Rate',
                    'Total_PnL', 'Average_PnL', 'Max_PnL', 'Min_PnL',
                    'Max_Drawdown', 'Sharpe_Ratio', 'Sortino_Ratio', 'Profit_Factor',
                    'Initial_Deposit', 'Final_Balance', 'Gross_Profit', 'Gross_Loss',
                    'Balance_Absolute_Drawdown'
                ],
                'Value': [
                    total_trades, winning_trades, losing_trades, win_rate,
                    total_pnl, average_pnl, max_pnl, min_pnl,
                    max_drawdown, sharpe_ratio, sortino_ratio, profit_factor,
                    self.balance, final_balance, gross_profit, -gross_loss,
                    max_drawdown
                ]
            })
            
        except Exception as e:
            logger.error(f"Error calculating evaluation metrics: {str(e)}")
            logger.error(f"Trades DataFrame columns: {self.Trades.columns.tolist()}")
            raise

    def _calculate_strategy_statistics(self) -> None:
        """Calculate strategy statistics using vectorized operations"""
        try:
            # Get basic metrics
            total_trades = self.metrics.loc[self.metrics['Metric'] == 'Total_Trades', 'Value'].values[0]
            initial_deposit = self.metrics.loc[self.metrics['Metric'] == 'Initial_Deposit', 'Value'].values[0]
            final_balance = self.metrics.loc[self.metrics['Metric'] == 'Final_Balance', 'Value'].values[0]
            total_net_profit = self.metrics.loc[self.metrics['Metric'] == 'Total_PnL', 'Value'].values[0]
            gross_profit = self.metrics.loc[self.metrics['Metric'] == 'Gross_Profit', 'Value'].values[0]
            gross_loss = abs(self.metrics.loc[self.metrics['Metric'] == 'Gross_Loss', 'Value'].values[0])
            max_drawdown = abs(self.metrics.loc[self.metrics['Metric'] == 'Max_Drawdown', 'Value'].values[0])
            
            # Calculate returns
            pnls = self.Trades['PnL'].values
            returns = pnls / initial_deposit if initial_deposit > 0 else np.zeros_like(pnls)
            
            # Calculate statistics
            stats_dict = {
                'Ticker': self.Ticker,
                'Profit_Factor': gross_profit / gross_loss if gross_loss > self.EPS else float('inf'),
                'Recovery_Factor': total_net_profit / max_drawdown if max_drawdown > self.EPS else 0.0,
                'AHPR': (final_balance - initial_deposit) / (initial_deposit * total_trades) if total_trades > 0 else 0,
                'GHPR': ((final_balance / initial_deposit) ** (1 / total_trades)) - 1 if total_trades > 0 and initial_deposit > 0 else 0,
                'Expected_Payoff': total_net_profit / total_trades if total_trades > 0 else 0,
                'Win_Loss_Ratio': np.sum(pnls > 0) / np.sum(pnls < 0) if np.sum(pnls < 0) > 0 else float('inf'),
                'Win_Rate': (np.sum(pnls > 0) / total_trades) * 100 if total_trades > 0 else 0
            }
            
            # Calculate risk-adjusted returns
            risk_free_rate = 0.0001  # 0.01% per trade
            avg_return = np.mean(returns)
            std_dev = np.std(returns)
            
            # Sharpe Ratio
            if abs(std_dev) > self.EPS and abs(avg_return - risk_free_rate) > self.EPS:
                stats_dict['Sharpe_Ratio'] = (avg_return - risk_free_rate) / std_dev
            else:
                stats_dict['Sharpe_Ratio'] = 0.0
                
            # Sortino Ratio
            downside_returns = np.clip(returns - risk_free_rate, a_min=None, a_max=0)
            downside_volatility = np.std(downside_returns, ddof=0)
            
            if abs(downside_volatility) > self.EPS and abs(avg_return - risk_free_rate) > self.EPS:
                stats_dict['Sortino_ratio'] = (avg_return - risk_free_rate) / downside_volatility
            else:
                stats_dict['Sortino_ratio'] = 0.0
                
            # Calmar Ratio
            equity_curve = self.Trades['Balance'].values
            returns = np.diff(equity_curve) / equity_curve[:-1] if len(equity_curve) > 1 else np.array([0])
            try:
                compounded_growth = np.prod(1 + returns) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
            except ZeroDivisionError:
                compounded_growth = 0
                
            max_drawdown = abs(np.max(np.maximum.accumulate(equity_curve) - equity_curve)) if len(equity_curve) > 0 else 0
            
            if max_drawdown > self.EPS:
                stats_dict['calmar_ratio'] = abs(compounded_growth) / max_drawdown
            else:
                stats_dict['calmar_ratio'] = 0.0
                
            # Convert to DataFrame
            self.stats = pd.DataFrame([stats_dict]).transpose().reset_index()
            self.stats.columns = ['Statistic', 'Value']
            
        except Exception as e:
            logger.error(f"Error calculating strategy statistics: {str(e)}")
            logger.error(f"Metrics DataFrame columns: {self.metrics.columns.tolist()}")
            raise

    def _interpret_trading_metrics(self) -> None:
        """Interpret trading metrics and provide recommendations"""
        # Define interpretation criteria
        interpretation_criteria = {
            'Ticker': lambda x: (x if x != '' else '', 'Symbol'),
            'Win_Loss_Ratio': lambda x: ('Good' if x > 0.5 else 'Poor', 'A high Win/Loss Ratio indicates a favorable risk-reward balance.'),
            'Win_Rate': lambda x: ('Good' if x > 50 else 'Poor', 'A high Win Rate suggests a high percentage of profitable trades.'),
            'Profit_Factor': lambda x: ('Good' if x > 1.5 else 'Poor', 'A Profit Factor greater than 1.5 indicates profitable trading.'),
            'Recovery_Factor': lambda x: ('Good' if x > 0.2 else 'Poor', 'A high Recovery Factor suggests effective recovery from drawdowns.'),
            'AHPR': lambda x: ('Good' if x > 0 else 'Poor', 'A positive AHPR indicates positive average holding period returns.'),
            'GHPR': lambda x: ('Good' if x > 0 else 'Poor', 'A positive GHPR indicates positive geometric holding period returns.'),
            'Expected_Payoff': lambda x: ('Good' if x > 0 else 'Poor', 'A positive Expected Payoff indicates favorable risk-reward per trade.'),
            'Sharpe_Ratio': lambda x: ('Good' if x > 1 else 'Poor', 'A Sharpe Ratio > 1 suggests good risk-adjusted returns.'),
            'Sortino_ratio': lambda x: ('Good' if x > 1 else 'Poor', 'A Sortino Ratio > 1 indicates good downside risk-adjusted returns.'),
            'calmar_ratio': lambda x: ('Good' if x > 0.5 else 'Poor', 'A Calmar Ratio > 0.5 indicates good drawdown-adjusted returns.')
        }
        
        # Add interpretation columns
        self.stats[['Interpretation', 'Detailed_Insights']] = self.stats.apply(
            lambda row: interpretation_criteria[row['Statistic']](row['Value']), 
            axis=1, 
            result_type='expand'
        )
        
        # Calculate weighted score
        important_metrics = ['Win_Rate', 'Profit_Factor', 'Sharpe_Ratio', 'Sortino_ratio']
        weights = {'Win_Rate': 0.3, 'Profit_Factor': 0.3, 'Sharpe_Ratio': 0.2, 'Sortino_ratio': 0.2}
        
        weighted_score = 0
        for metric in important_metrics:
            if metric in self.stats['Statistic'].values:
                interpretation = self.stats.loc[self.stats['Statistic'] == metric, 'Interpretation'].values[0]
                weighted_score += weights[metric] * (1 if interpretation == 'Good' else 0)
                
        # Set final recommendation
        self.Final_Recommendation = 'Consider for Trading' if weighted_score >= 0.6 else 'Do Not Consider for Trading' 