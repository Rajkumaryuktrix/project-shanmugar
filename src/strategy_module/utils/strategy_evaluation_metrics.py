import numpy as np
from typing import Dict, List, Optional, Union
import logging
from dataclasses import dataclass
import json
from datetime import datetime

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclass
class TradeMetrics:
    """Data class to store trade metrics"""
    trade_id: int
    open_action: str
    close_action: str
    trade_type: str
    open_time: int
    close_time: int
    volume: float
    open_price: float
    close_price: float
    pnl: float
    mae: float
    mfe: float
    achieved_rr: float
    opportunity_rr: float
    duration: float
    capital_invested: float
    capital_risked_pct: float
    roi: float
    account_balance: float
    commission: float

class StrategyEvaluationMetrics:
    def __init__(self, backtest_results: Dict, ticker: str):
        """
        Initialize strategy evaluation metrics.
        
        Args:
            backtest_results: Dictionary containing backtest results
            ticker: Ticker symbol
        """
        self.ticker = ticker
        self.trades = backtest_results['trades']
        self.balance_history = np.array(backtest_results['balance_history'])
        self.equity_curve = np.array(backtest_results['equity_curve'])
        self.drawdown_curve = np.array(backtest_results['drawdown_curve'])
        
        # Get initial balance from balance history
        self.initial_balance = float(self.balance_history[0]) if len(self.balance_history) > 0 else 0.0
        
        # Initialize metrics
        self.metrics = {}
        self.statistics = {}
        self.recommendation = None
        
        # Calculate metrics
        self._calculate_metrics()

    def _calculate_metrics(self) -> None:
        """Calculate all strategy evaluation metrics"""
        try:
            if not self.trades:
                self._set_empty_metrics()
                return

            # Calculate basic trade metrics
            self._calculate_trade_metrics()
            
            # Calculate drawdown metrics
            self._calculate_drawdown_metrics()
            
            # Calculate risk-adjusted returns
            self._calculate_risk_adjusted_returns()
            
            # Calculate CAGR and Calmar ratio
            self._calculate_cagr_and_calmar()
            
            # Calculate trade statistics
            self._calculate_trade_statistics()
            
            # Generate recommendation
            self._generate_recommendation()

        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            self._set_empty_metrics()

    def _calculate_trade_metrics(self) -> None:
        """Calculate basic trade metrics"""
        try:
            if not self.trades:
                return
                
            # Calculate win rate and profit metrics
            winning_trades = [t for t in self.trades if float(t['PnL']) > 0]
            losing_trades = [t for t in self.trades if float(t['PnL']) < 0]
            
            total_trades = len(self.trades)
            winning_trades_count = len(winning_trades)
            losing_trades_count = len(losing_trades)
            
            win_rate = winning_trades_count / total_trades if total_trades > 0 else 0.0
            
            gross_profit = sum(float(t['PnL']) for t in winning_trades)
            gross_loss = abs(sum(float(t['PnL']) for t in losing_trades))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            avg_win = np.mean([float(t['PnL']) for t in winning_trades]) if winning_trades else 0.0
            avg_loss = np.mean([abs(float(t['PnL'])) for t in losing_trades]) if losing_trades else 0.0
            
            expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
            
            self.metrics.update({
                'win_rate': win_rate * 100,  # Convert to percentage
                'profit_factor': profit_factor,
                'expectancy': expectancy,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'total_trades': total_trades,
                'winning_trades': winning_trades_count,
                'losing_trades': losing_trades_count
            })
            
        except Exception as e:
            logger.error(f"Error calculating trade metrics: {str(e)}")

    def _calculate_drawdown_metrics(self) -> None:
        """Calculate drawdown metrics"""
        try:
            if len(self.equity_curve) > 1:
                peak = np.maximum.accumulate(self.equity_curve)
                drawdown = np.where(peak > 0, (peak - self.equity_curve) / peak * 100, 0)
                max_drawdown = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0
                avg_drawdown = float(np.mean(drawdown)) if len(drawdown) > 0 else 0.0
            else:
                max_drawdown = 0.0
                avg_drawdown = 0.0
                
            self.metrics.update({
                'max_drawdown': max_drawdown,
                'avg_drawdown': avg_drawdown
            })
            
        except Exception as e:
            logger.error(f"Error calculating drawdown metrics: {str(e)}")

    def _calculate_risk_adjusted_returns(self) -> None:
        """Calculate risk-adjusted return metrics"""
        try:
            if len(self.equity_curve) > 1:
                # Calculate daily returns (log returns for better statistical properties)
                daily_returns = np.diff(np.log(self.equity_curve)) * 100
                
                # Calculate annualized metrics
                mean_return = np.mean(daily_returns) if len(daily_returns) > 0 else 0.0
                std_return = np.std(daily_returns) if len(daily_returns) > 1 else 0.0
                negative_returns = daily_returns[daily_returns < 0]
                
                # Calculate Sharpe ratio (annualized)
                risk_free_rate = 0.02  # 2% risk-free rate
                excess_returns = mean_return - (risk_free_rate / 252)  # Daily risk-free rate
                sharpe = np.sqrt(252) * excess_returns / std_return if std_return > 0 else 0.0
                
                # Calculate Sortino ratio (annualized)
                downside_deviation = np.sqrt(np.mean(negative_returns ** 2)) if len(negative_returns) > 0 else 0.0
                sortino = np.sqrt(252) * excess_returns / downside_deviation if downside_deviation > 0 else 0.0
                
                # Calculate Calmar ratio
                calmar = excess_returns * 252 / (self.metrics.get('max_drawdown', 0.0) / 100) if self.metrics.get('max_drawdown', 0.0) > 0 else 0.0
            else:
                sharpe = 0.0
                sortino = 0.0
                calmar = 0.0
                
            self.metrics.update({
                'sharpe': sharpe,
                'sortino': sortino,
                'calmar': calmar
            })
            
        except Exception as e:
            logger.error(f"Error calculating risk-adjusted returns: {str(e)}")

    def _calculate_cagr_and_calmar(self) -> None:
        """Calculate CAGR and Calmar ratio"""
        try:
            if len(self.equity_curve) > 1:
                initial_balance = float(self.equity_curve[0])
                final_balance = float(self.equity_curve[-1])
                trading_days = float(self.trades[-1]['Exit_Time'] - self.trades[0]['Entry_Time']) / (24 * 3600)
                
                if trading_days > 0 and initial_balance > 0:
                    cagr = ((final_balance / initial_balance) ** (252 / trading_days) - 1) * 100
                else:
                    cagr = 0.0
                    final_balance = self.initial_balance
            else:
                cagr = 0.0
                final_balance = self.initial_balance
                
            # Calculate Calmar ratio
            if self.metrics.get('max_drawdown', 0.0) > 0 and cagr > 0:
                raw_calmar = cagr / self.metrics['max_drawdown']
                calmar_ratio = min(raw_calmar, 100)  # Cap at 100
            else:
                calmar_ratio = 0.0
                
            self.metrics.update({
                'cagr': cagr,
                'calmar_ratio': calmar_ratio,
                'final_balance': final_balance
            })
            
        except Exception as e:
            logger.error(f"Error calculating CAGR and Calmar: {str(e)}")

    def _calculate_trade_statistics(self) -> None:
        """Calculate trade statistics"""
        try:
            if not self.trades:
                return
                
            # Calculate average trade duration
            avg_trade_duration = np.mean([float(t.get('Duration', 0)) for t in self.trades]) if self.trades and 'Duration' in self.trades[0] else 0.0
            
            # Calculate net profit and commission
            net_profit = sum(float(t['PnL']) for t in self.trades)
            total_commission = sum(float(t['Commission']) for t in self.trades)
            
            self.metrics.update({
                'avg_trade_duration': avg_trade_duration,
                'net_profit': net_profit,
                'total_commission': total_commission
            })
            
            # Format statistics for display
            self.statistics = {
                'win_rate': f"{self.metrics.get('win_rate', 0.0):.2f}%",
                'profit_factor': f"{self.metrics.get('profit_factor', 0.0):.2f}",
                'sharpe_ratio': f"{self.metrics.get('sharpe', 0.0):.2f}",
                'sortino_ratio': f"{self.metrics.get('sortino', 0.0):.2f}",
                'calmar_ratio': f"{self.metrics.get('calmar', 0.0):.2f}",
                'max_drawdown': f"{self.metrics.get('max_drawdown', 0.0):.2f}%",
                'cagr': f"{self.metrics.get('cagr', 0.0):.2f}%",
                'total_trades': str(self.metrics.get('total_trades', 0)),
                'winning_trades': str(self.metrics.get('winning_trades', 0)),
                'losing_trades': str(self.metrics.get('losing_trades', 0)),
                'net_profit': f"${net_profit:.2f}",
                'total_commission': f"${total_commission:.2f}",
                'final_balance': f"${self.metrics.get('final_balance', self.initial_balance):.2f}",
                'initial_balance': f"${self.initial_balance:.2f}"
            }
            
        except Exception as e:
            logger.error(f"Error calculating trade statistics: {str(e)}")

    def _set_empty_metrics(self) -> None:
        """Set empty metrics when no trades are available"""
        self.metrics = {
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'sharpe': 0.0,
            'sortino': 0.0,
            'calmar': 0.0,
            'max_drawdown': 0.0,
            'cagr': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'final_balance': self.initial_balance
        }
        
        self.statistics = {
            'win_rate': '0.00%',
            'profit_factor': '0.00',
            'sharpe_ratio': '0.00',
            'sortino_ratio': '0.00',
            'calmar_ratio': '0.00',
            'max_drawdown': '0.00%',
            'cagr': '0.00%',
            'total_trades': '0',
            'winning_trades': '0',
            'losing_trades': '0',
            'net_profit': '$0.00',
            'total_commission': '$0.00',
            'final_balance': f"${self.initial_balance:.2f}",
            'initial_balance': f"${self.initial_balance:.2f}"
        }
        self.recommendation = "No trades available for evaluation"

    def _generate_recommendation(self) -> str:
        """Generate strategy recommendation based on metrics"""
        if self.metrics['total_trades'] < 30:
            return "Insufficient trades for reliable evaluation"
            
        recommendations = []
        
        # Win rate analysis
        if self.metrics['win_rate'] >= 60:
            recommendations.append("Strong win rate")
        elif self.metrics['win_rate'] >= 50:
            recommendations.append("Moderate win rate")
        else:
            recommendations.append("Low win rate")
            
        # Profit factor analysis
        if self.metrics['profit_factor'] >= 2:
            recommendations.append("Excellent profit factor")
        elif self.metrics['profit_factor'] >= 1.5:
            recommendations.append("Good profit factor")
        else:
            recommendations.append("Poor profit factor")
            
        # Risk-adjusted returns
        sharpe = self.metrics.get('sharpe', 0.0)
        if sharpe >= 2:
            recommendations.append("Excellent risk-adjusted returns")
        elif sharpe >= 1:
            recommendations.append("Good risk-adjusted returns")
        else:
            recommendations.append("Poor risk-adjusted returns")
            
        # Drawdown analysis
        if self.metrics['max_drawdown'] <= 10:
            recommendations.append("Excellent drawdown control")
        elif self.metrics['max_drawdown'] <= 20:
            recommendations.append("Good drawdown control")
        else:
            recommendations.append("Poor drawdown control")
            
        # Overall recommendation
        if (self.metrics['win_rate'] >= 50 and 
            self.metrics['profit_factor'] >= 1.5 and 
            sharpe >= 1 and 
            self.metrics['max_drawdown'] <= 20):
            return f"STRONG BUY: {'; '.join(recommendations)}"
        elif (self.metrics['win_rate'] >= 45 and 
              self.metrics['profit_factor'] >= 1.2 and 
              sharpe >= 0.8 and 
              self.metrics['max_drawdown'] <= 25):
            return f"BUY: {'; '.join(recommendations)}"
        else:
            return f"NEUTRAL: {'; '.join(recommendations)}"
    
    def get_metrics(self) -> Dict:
        """Get calculated metrics"""
        return self.metrics

    def get_statistics(self) -> Dict:
        """Get formatted statistics"""
        return self.statistics

    def get_recommendation(self) -> str:
        """Get strategy recommendation"""
        return self.recommendation
    
    def save_results(self, filepath: str) -> None:
        """Save evaluation results to JSON file"""
        results = {
            'ticker': self.ticker,
            'metrics': self.metrics,
            'statistics': self.statistics,
            'recommendation': self.recommendation,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4) 