import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Union
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UpstoxHistoricalData:
    def __init__(self):
        """
        Initialize the Upstox Historical Data client.
        
        Args:
            access_token (str): Upstox API access token
        """
        load_dotenv()
        self.access_token = os.getenv('UPSTOX_ACCESS_TOKEN')
        if not self.access_token:
            logger.error("UPSTOX_ACCESS_TOKEN not found in .env file")
            raise ValueError("Please set UPSTOX_ACCESS_TOKEN in your .env file")
    

        self.base_url = "https://api.upstox.com"
        self.headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.access_token}'
        }
        
        # Define valid units and their interval constraints
        self.unit_constraints = {
            'minutes': {
                'valid_intervals': range(1, 301),  # 1 to 300
                'max_records': {
                    '1-15': 30,  # 1 month for 1-15 minute intervals
                    '16-300': 90  # 1 quarter for >15 minute intervals
                }
            },
            'hours': {
                'valid_intervals': range(1, 6),  # 1 to 5
                'max_records': 90  # 1 quarter
            },
            'days': {
                'valid_intervals': [1],
                'max_records': 3650  # 1 decade
            },
            'weeks': {
                'valid_intervals': [1],
                'max_records': None  # No limit
            },
            'months': {
                'valid_intervals': [1],
                'max_records': None  # No limit
            }
        }

    def _validate_dates(self, from_date: str, to_date: str) -> tuple:
        """
        Validate and parse dates.
        
        Args:
            from_date (str): Start date in YYYY-MM-DD format
            to_date (str): End date in YYYY-MM-DD format
            
        Returns:
            tuple: (from_dt, to_dt) as datetime objects
            
        Raises:
            ValueError: If dates are invalid
        """
        try:
            from_dt = datetime.strptime(from_date, '%Y-%m-%d')
            to_dt = datetime.strptime(to_date, '%Y-%m-%d')
            
            if from_dt > to_dt:
                raise ValueError("from_date must be before to_date")
                
            current_date = datetime.now()
            if from_dt > current_date or to_dt > current_date:
                raise ValueError("Cannot fetch data for future dates")
                
            return from_dt, to_dt
            
        except ValueError as e:
            logger.error(f"Date validation error: {str(e)}")
            raise

    def _validate_unit_interval(self, unit: str, interval: int) -> None:
        """
        Validate unit and interval combination.
        
        Args:
            unit (str): Time unit (minutes, hours, days, weeks, months)
            interval (int): Time interval
            
        Raises:
            ValueError: If unit or interval is invalid
        """
        if unit not in self.unit_constraints:
            raise ValueError(f"Invalid unit. Must be one of: {list(self.unit_constraints.keys())}")
            
        if interval not in self.unit_constraints[unit]['valid_intervals']:
            raise ValueError(f"Invalid interval for {unit}. Valid intervals: {list(self.unit_constraints[unit]['valid_intervals'])}")

    def _adjust_date_range(self, from_dt: datetime, to_dt: datetime, unit: str, interval: int) -> datetime:
        """
        Adjust date range based on unit and interval constraints.
        
        Args:
            from_dt (datetime): Start date
            to_dt (datetime): End date
            unit (str): Time unit
            interval (int): Time interval
            
        Returns:
            datetime: Adjusted start date
        """
        max_days = self.unit_constraints[unit]['max_records']
        
        if max_days is None:
            return from_dt
            
        if unit == 'minutes':
            if interval <= 15:
                max_days = max_days['1-15']
            else:
                max_days = max_days['16-300']
                
        date_diff = (to_dt - from_dt).days
        if date_diff > max_days:
            logger.warning(f"Date range exceeds {max_days} days limit. Adjusting start date.")
            return to_dt - timedelta(days=max_days)
            
        return from_dt

    def get_historical_candles(
        self,
        instrument_key: str,
        unit: str,
        interval: int,
        to_date: str,
        from_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch historical candle data from Upstox API.
        
        Args:
            instrument_key (str): Instrument key for the symbol
            unit (str): Time unit (minutes, hours, days, weeks, months)
            interval (int): Time interval
            to_date (str): End date in YYYY-MM-DD format
            from_date (str, optional): Start date in YYYY-MM-DD format
            
        Returns:
            pd.DataFrame: Historical candle data with columns:
                - timestamp: Start time of the candle
                - open: Opening price
                - high: Highest price
                - low: Lowest price
                - close: Closing price
                - volume: Trading volume
                - oi: Open Interest
                
        Raises:
            ValueError: If parameters are invalid
            requests.exceptions.RequestException: If API request fails
        """
        try:
            # Validate unit and interval
            self._validate_unit_interval(unit, interval)
            
            # Validate and parse dates
            to_dt = datetime.strptime(to_date, '%Y-%m-%d')
            if from_date:
                from_dt = datetime.strptime(from_date, '%Y-%m-%d')
            else:
                # If from_date not provided, use appropriate default based on unit
                if unit == 'minutes':
                    from_dt = to_dt - timedelta(days=1)  # Default to 1 day for minutes
                else:
                    from_dt = to_dt - timedelta(days=30)  # Default to 30 days for other units
                    
            # Adjust date range if needed
            from_dt = self._adjust_date_range(from_dt, to_dt, unit, interval)
            from_date = from_dt.strftime('%Y-%m-%d')
            
            # Construct API URL
            url = f"{self.base_url}/v3/historical-candle/{instrument_key}/{unit}/{interval}/{to_date}/{from_date}"
            
            # Make API request
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') == 'error':
                raise ValueError(f"API Error: {data.get('message', 'Unknown error')}")
                
            # Process candle data
            candles = data.get('data', {}).get('candles', [])
            if not candles:
                logger.warning("No candle data found in the response")
                return pd.DataFrame()
                
            # Convert to DataFrame
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return candles
            # df.sort_values('timestamp')
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            if hasattr(e.response, 'text'):
                logger.error(f"API response: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            raise

    # Retrive Current Day Intraday candle data
    def get_intraday_candles(
        self,
        instrument_key: str,
        unit: str,
        interval: int) -> pd.DataFrame:
        """
        Fetch intraday candle data for the current trading day from Upstox API.
        
        Args:
            instrument_key (str): Instrument key for the symbol
            unit (str): Time unit (minutes, hours, days)
            interval (int): Time interval
                - For minutes: 1 to 300
                - For hours: 1 to 5
                - For days: 1
                
        Returns:
            pd.DataFrame: Intraday candle data with columns:
                - timestamp: Start time of the candle
                - open: Opening price
                - high: Highest price
                - low: Lowest price
                - close: Closing price
                - volume: Trading volume
                - oi: Open Interest
                
        Raises:
            ValueError: If parameters are invalid
            requests.exceptions.RequestException: If API request fails
        """
        try:
            # Define intraday-specific unit constraints
            intraday_constraints = {
                'minutes': {
                    'valid_intervals': range(1, 301),  # 1 to 300
                    'description': '1 to 300 minutes'
                },
                'hours': {
                    'valid_intervals': range(1, 6),  # 1 to 5
                    'description': '1 to 5 hours'
                },
                'days': {
                    'valid_intervals': [1],
                    'description': '1 day'
                }
            }
            
            # Validate unit
            if unit not in intraday_constraints:
                raise ValueError(
                    f"Invalid unit '{unit}'. Must be one of: {list(intraday_constraints.keys())}"
                )
                
            # Validate interval
            if interval not in intraday_constraints[unit]['valid_intervals']:
                raise ValueError(
                    f"Invalid interval {interval} for {unit}. "
                    f"Valid intervals: {intraday_constraints[unit]['description']}"
                )
                
            # Construct API URL
            url = f"{self.base_url}/v3/historical-candle/intraday/{instrument_key}/{unit}/{interval}"
            
            # Make API request
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') == 'error':
                raise ValueError(f"API Error: {data.get('message', 'Unknown error')}")
                
            # Process candle data
            candles = data.get('data', {}).get('candles', [])
            if not candles:
                logger.warning("No intraday candle data found in the response")
                return pd.DataFrame()
                
            # Convert to DataFrame
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return candles
            # return df.sort_values('timestamp')
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            if hasattr(e.response, 'text'):
                logger.error(f"API response: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error fetching intraday data: {str(e)}")
            raise


# Example usage
if __name__ == "__main__":

    client = UpstoxHistoricalData()
    
    try:
        # Example 1: Get historical 5-minute candles
        historical_df = client.get_historical_candles(
            instrument_key="NSE_FO|36702",
            unit="minutes",
            interval=5,
            to_date="2024-05-20",
            from_date="2024-04-01"
        )
        
        print("\nHistorical Candle Data:")
        print(historical_df)
#       print(historical_df.head())
        print(f"\nTotal historical candles: {len(historical_df)}")
        
        # Example 2: Get intraday 5-minute candles
        intraday_df = client.get_intraday_candles(
            instrument_key="NSE_FO|36702",
            unit="minutes",
            interval=5
        )
        
        print("\nIntraday Candle Data:")
        print(intraday_df.head())
        print(f"\nTotal intraday candles: {len(intraday_df)}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
