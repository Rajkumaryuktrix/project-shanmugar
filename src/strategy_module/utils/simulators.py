import numpy as np
from typing import Dict, List, Optional, Any, TYPE_CHECKING
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from .backward_testing import CandleData

if TYPE_CHECKING:
    from .forward_testing import ForwardConfig

logger = logging.getLogger(__name__)

@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation."""
    # Simulation parameters
    num_paths: int = 1000
    forecast_periods: int = 252  # Default to 1 year of trading days
    volatility_window: int = 20  # Window for volatility calculation
    drift_window: int = 20      # Window for drift calculation
    
    # Price parameters
    use_historical_vol: bool = True
    use_historical_drift: bool = True
    min_price: float = 0.0
    max_price: float = float('inf')
    
    # Volume parameters
    volume_scaling: float = 1.0
    min_volume: float = 0.0
    max_volume: float = float('inf')
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.num_paths <= 0:
            raise ValueError("Number of paths must be positive")
        if self.forecast_periods <= 0:
            raise ValueError("Forecast periods must be positive")
        if self.volatility_window <= 0:
            raise ValueError("Volatility window must be positive")
        if self.drift_window <= 0:
            raise ValueError("Drift window must be positive")
        if self.min_price < 0:
            raise ValueError("Minimum price cannot be negative")
        if self.max_price <= self.min_price:
            raise ValueError("Maximum price must be greater than minimum price")
        if self.volume_scaling <= 0:
            raise ValueError("Volume scaling must be positive")
        if self.min_volume < 0:
            raise ValueError("Minimum volume cannot be negative")
        if self.max_volume <= self.min_volume:
            raise ValueError("Maximum volume must be greater than minimum volume")

@dataclass
class BrownianMotionConfig:
    """Configuration for Brownian Motion simulation."""
    # Simulation parameters
    num_paths: int = 1000
    forecast_periods: int = 252  # Default to 1 year of trading days
    volatility: float = 0.2     # Annualized volatility
    drift: float = 0.0         # Annualized drift
    
    # Price parameters
    min_price: float = 0.0
    max_price: float = float('inf')
    
    # Volume parameters
    volume_scaling: float = 1.0
    min_volume: float = 0.0
    max_volume: float = float('inf')
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.num_paths <= 0:
            raise ValueError("Number of paths must be positive")
        if self.forecast_periods <= 0:
            raise ValueError("Forecast periods must be positive")
        if self.volatility <= 0:
            raise ValueError("Volatility must be positive")
        if self.min_price < 0:
            raise ValueError("Minimum price cannot be negative")
        if self.max_price <= self.min_price:
            raise ValueError("Maximum price must be greater than minimum price")
        if self.volume_scaling <= 0:
            raise ValueError("Volume scaling must be positive")
        if self.min_volume < 0:
            raise ValueError("Minimum volume cannot be negative")
        if self.max_volume <= self.min_volume:
            raise ValueError("Maximum volume must be greater than minimum volume")

class BaseSimulator(ABC):
    """Abstract base class for all simulators."""
    
    def __init__(self, config: 'ForwardConfig'):
        self.config = config
        self.validate_config()
    
    @abstractmethod
    def validate_config(self) -> None:
        """Validate simulator-specific configuration."""
        pass
    
    @abstractmethod
    def generate_paths(self) -> Dict[str, CandleData]:
        """Generate simulation paths."""
        pass
    
    @abstractmethod
    def run_simulation(self) -> Dict[str, Any]:
        """Run the simulation."""
        pass
    
    def get_name(self) -> str:
        """Get simulator name."""
        return self.__class__.__name__

class MonteCarloSimulator(BaseSimulator):
    """Monte Carlo simulator for generating price paths."""
    
    def __init__(self, config: 'ForwardConfig', mc_config: Optional[MonteCarloConfig] = None):
        """
        Initialize Monte Carlo simulator.
        
        Args:
            config: Forward testing configuration
            mc_config: Monte Carlo specific configuration
        """
        self.mc_config = mc_config or MonteCarloConfig()
        self.mc_config.validate()
        super().__init__(config)
        
    def validate_config(self) -> None:
        """Validate simulator-specific configuration."""
        if not isinstance(self.config.candle_data, CandleData):
            raise ValueError("CandleData must be provided for Monte Carlo simulation")
        if len(self.config.candle_data.close) < self.mc_config.volatility_window:
            raise ValueError(f"Insufficient data for volatility calculation. Need at least {self.mc_config.volatility_window} periods")
            
    def _calculate_historical_volatility(self) -> float:
        """Calculate historical volatility from price data."""
        returns = np.diff(np.log(self.config.candle_data.close))
        return np.std(returns) * np.sqrt(252)  # Annualized volatility
        
    def _calculate_historical_drift(self) -> float:
        """Calculate historical drift from price data."""
        returns = np.diff(np.log(self.config.candle_data.close))
        return np.mean(returns) * 252  # Annualized drift
        
    def _generate_price_path(self, start_price: float, volatility: float, drift: float) -> np.ndarray:
        """Generate a single price path using geometric Brownian motion."""
        dt = 1/252  # Daily time step
        t = np.arange(self.mc_config.forecast_periods)
        
        # Generate random walk
        random_walk = np.random.normal(0, 1, self.mc_config.forecast_periods)
        
        # Calculate drift and volatility terms
        drift_term = (drift - 0.5 * volatility**2) * t * dt
        volatility_term = volatility * np.sqrt(dt) * np.cumsum(random_walk)
        
        # Generate price path
        price_path = start_price * np.exp(drift_term + volatility_term)
        
        # Apply price bounds
        price_path = np.clip(price_path, self.mc_config.min_price, self.mc_config.max_price)
        
        return price_path
        
    def _generate_volume_path(self, base_volume: float) -> np.ndarray:
        """Generate volume path based on historical patterns."""
        # Use historical volume as base and add some randomness
        volume_std = np.std(self.config.candle_data.volume) * self.mc_config.volume_scaling
        volume_path = np.random.normal(base_volume, volume_std, self.mc_config.forecast_periods)
        
        # Apply volume bounds
        volume_path = np.clip(volume_path, self.mc_config.min_volume, self.mc_config.max_volume)
        
        return volume_path
        
    def generate_paths(self) -> Dict[str, CandleData]:
        """Generate multiple price paths using Monte Carlo simulation."""
        try:
            # Calculate simulation parameters
            start_price = self.config.candle_data.close[-1]
            volatility = self._calculate_historical_volatility() if self.mc_config.use_historical_vol else 0.2
            drift = self._calculate_historical_drift() if self.mc_config.use_historical_drift else 0.0
            base_volume = np.mean(self.config.candle_data.volume[-self.mc_config.volatility_window:])
            
            # Generate timestamps
            last_timestamp = self.config.candle_data.timestamp[-1]
            timestamps = np.array([
                int((datetime.fromtimestamp(last_timestamp) + timedelta(days=i)).timestamp())
                for i in range(1, self.mc_config.forecast_periods + 1)
            ])
            
            paths = {}
            for i in range(self.mc_config.num_paths):
                # Generate price path
                price_path = self._generate_price_path(start_price, volatility, drift)
                
                # Generate volume path
                volume_path = self._generate_volume_path(base_volume)
                
                # Create CandleData object for this path
                path_data = CandleData(
                    timestamp=timestamps,
                    open=price_path,
                    high=price_path * (1 + np.random.uniform(0, 0.02, self.mc_config.forecast_periods)),
                    low=price_path * (1 - np.random.uniform(0, 0.02, self.mc_config.forecast_periods)),
                    close=price_path,
                    volume=volume_path,
                    oi=np.zeros(self.mc_config.forecast_periods),  # Placeholder for OI
                    signal=np.zeros(self.mc_config.forecast_periods)  # No signals in simulation
                )
                
                paths[f'mc_path_{i+1}'] = path_data
                
            logger.info(f"Generated {self.mc_config.num_paths} Monte Carlo paths")
            return paths
            
        except Exception as e:
            logger.error(f"Error generating Monte Carlo paths: {str(e)}")
            raise
            
    def run_simulation(self) -> Dict[str, Any]:
        """Run Monte Carlo simulation and return results."""
        try:
            paths = self.generate_paths()
            return {
                'paths': paths,
                'config': {
                    'num_paths': self.mc_config.num_paths,
                    'forecast_periods': self.mc_config.forecast_periods,
                    'volatility_window': self.mc_config.volatility_window,
                    'drift_window': self.mc_config.drift_window,
                    'use_historical_vol': self.mc_config.use_historical_vol,
                    'use_historical_drift': self.mc_config.use_historical_drift
                }
            }
        except Exception as e:
            logger.error(f"Error in Monte Carlo simulation: {str(e)}")
            raise

class BrownianMotionSimulator(BaseSimulator):
    """Brownian Motion simulator for generating price paths."""
    
    def __init__(self, config: 'ForwardConfig', bm_config: Optional[BrownianMotionConfig] = None):
        """
        Initialize Brownian Motion simulator.
        
        Args:
            config: Forward testing configuration
            bm_config: Brownian Motion specific configuration
        """
        self.bm_config = bm_config or BrownianMotionConfig()
        self.bm_config.validate()
        super().__init__(config)
        
    def validate_config(self) -> None:
        """Validate simulator-specific configuration."""
        if not isinstance(self.config.candle_data, CandleData):
            raise ValueError("CandleData must be provided for Brownian Motion simulation")
        if len(self.config.candle_data.close) < 2:
            raise ValueError("Insufficient data for Brownian Motion simulation. Need at least 2 periods")
            
    def _generate_price_path(self, start_price: float) -> np.ndarray:
        """Generate a single price path using standard Brownian motion."""
        dt = 1/252  # Daily time step
        t = np.arange(self.bm_config.forecast_periods)
        
        # Generate random walk
        random_walk = np.random.normal(0, 1, self.bm_config.forecast_periods)
        
        # Calculate drift and volatility terms
        drift_term = self.bm_config.drift * t * dt
        volatility_term = self.bm_config.volatility * np.sqrt(dt) * np.cumsum(random_walk)
        
        # Generate price path
        price_path = start_price + drift_term + volatility_term
        
        # Apply price bounds
        price_path = np.clip(price_path, self.bm_config.min_price, self.bm_config.max_price)
        
        return price_path
        
    def _generate_volume_path(self, base_volume: float) -> np.ndarray:
        """Generate volume path based on historical patterns."""
        # Use historical volume as base and add some randomness
        volume_std = np.std(self.config.candle_data.volume) * self.bm_config.volume_scaling
        volume_path = np.random.normal(base_volume, volume_std, self.bm_config.forecast_periods)
        
        # Apply volume bounds
        volume_path = np.clip(volume_path, self.bm_config.min_volume, self.bm_config.max_volume)
        
        return volume_path
        
    def generate_paths(self) -> Dict[str, CandleData]:
        """Generate multiple price paths using Brownian Motion simulation."""
        try:
            # Calculate simulation parameters
            start_price = self.config.candle_data.close[-1]
            base_volume = np.mean(self.config.candle_data.volume[-20:])  # Use last 20 periods for volume
            
            # Generate timestamps
            last_timestamp = self.config.candle_data.timestamp[-1]
            timestamps = np.array([
                int((datetime.fromtimestamp(last_timestamp) + timedelta(days=i)).timestamp())
                for i in range(1, self.bm_config.forecast_periods + 1)
            ])
            
            paths = {}
            for i in range(self.bm_config.num_paths):
                # Generate price path
                price_path = self._generate_price_path(start_price)
                
                # Generate volume path
                volume_path = self._generate_volume_path(base_volume)
                
                # Create CandleData object for this path
                path_data = CandleData(
                    timestamp=timestamps,
                    open=price_path,
                    high=price_path * (1 + np.random.uniform(0, 0.02, self.bm_config.forecast_periods)),
                    low=price_path * (1 - np.random.uniform(0, 0.02, self.bm_config.forecast_periods)),
                    close=price_path,
                    volume=volume_path,
                    oi=np.zeros(self.bm_config.forecast_periods),  # Placeholder for OI
                    signal=np.zeros(self.bm_config.forecast_periods)  # No signals in simulation
                )
                
                paths[f'bm_path_{i+1}'] = path_data
                
            logger.info(f"Generated {self.bm_config.num_paths} Brownian Motion paths")
            return paths
            
        except Exception as e:
            logger.error(f"Error generating Brownian Motion paths: {str(e)}")
            raise
            
    def run_simulation(self) -> Dict[str, Any]:
        """Run Brownian Motion simulation and return results."""
        try:
            paths = self.generate_paths()
            return {
                'paths': paths,
                'config': {
                    'num_paths': self.bm_config.num_paths,
                    'forecast_periods': self.bm_config.forecast_periods,
                    'volatility': self.bm_config.volatility,
                    'drift': self.bm_config.drift
                }
            }
        except Exception as e:
            logger.error(f"Error in Brownian Motion simulation: {str(e)}")
            raise 