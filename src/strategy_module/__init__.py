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

import logging
from pathlib import Path
import json
import os

# Setup logging
logger = logging.getLogger(__name__)

# Define paths
MODULE_PATH = Path(__file__).parent
RESULTS_PATH = MODULE_PATH / 'results'
CONFIG_PATH = MODULE_PATH / 'config'

# Create necessary directories
RESULTS_PATH.mkdir(exist_ok=True)
CONFIG_PATH.mkdir(exist_ok=True)

# Import strategies
from .strategies.rsi_strategy import RSIStrategy
from .strategies.moving_average_strategy import MovingAverageStrategy
from .strategies.time_based_strategy import TimeBasedStrategy

# Import core components
from .core_engine import StrategyEngine
from .strategy_cli import StrategyCLI
from .utils import (
    StrategyTester,
    BacktestingEngine,
    ForwardTestingEngine,
    AdvancedStatisticalTesting,
    StrategyEvaluationMetrics
)

__all__ = [
    # Core Components
    'StrategyEngine',
    'StrategyCLI',
    
    # Strategy Implementations
    'RSIStrategy',
    'MovingAverageStrategy',
    'TimeBasedStrategy',
    
    # Testing and Evaluation Tools
    'StrategyTester',
    'BacktestingEngine',
    'ForwardTestingEngine',
    'AdvancedStatisticalTesting',
    'StrategyEvaluationMetrics'
]

__version__ = '1.0.0' 