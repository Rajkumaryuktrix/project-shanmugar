"""
Strategy Module for Algorithmic Trading System

This module provides the core functionality for implementing, testing, and executing
trading strategies. It includes strategy implementations, backtesting capabilities,
and strategy evaluation tools.

Components:
    - Core Engine: Main strategy execution engine
    - Strategies: Various trading strategy implementations
    - Utils: Testing and evaluation utilities
    - CLI: Command-line interface for strategy management
"""

from .core_engine import StrategyEngine
from .strategy_cli import StrategyCLI
from .utils import (
    StrategyTester,
    BacktestingTradeEngine,
    ForwardTesting,
    AdvancedStatisticalTesting,
    StrategyEvaluationMetrics
)

# Import available strategies
from .strategies.rsi_strategy import RSIStrategy
from .strategies.moving_average_strategy import MovingAverageCrossoverStrategy

__all__ = [
    # Core Components
    'StrategyEngine',
    'StrategyCLI',
    
    # Strategy Implementations
    'RSIStrategy',
    'MovingAverageCrossoverStrategy',
    
    # Testing and Evaluation Tools
    'StrategyTester',
    'BacktestingTradeEngine',
    'ForwardTesting',
    'AdvancedStatisticalTesting',
    'StrategyEvaluationMetrics'
]

__version__ = '1.0.0' 