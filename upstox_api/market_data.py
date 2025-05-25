import requests
import os
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging
from typing import Dict, List, Optional, Union
from .symbol_codes import get_instrument_key, check_symbol_details, get_available_symbols, SYMBOL_TO_ISIN
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UpstoxMarketData:
    """
    A class to handle all market data operations using Upstox API.
    Includes functionality for:
    - Real-time market quotes
    - Historical data
    - Intraday data
    - OHLC data
    - LTP quotes
    - Option Greeks
    - Market depth
    """
    
    def __init__(self):
        """Initialize the market data manager with API credentials."""
        load_dotenv()
        self.access_token = os.getenv('UPSTOX_ACCESS_TOKEN')
        if not self.access_token:
            raise ValueError("UPSTOX_ACCESS_TOKEN not found in environment variables")
        
        self.base_url = "https://api.upstox.com"
        self.headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.access_token}'
        }
        
        # Timeframe mapping for historical and intraday data
        self.timeframe_map = {
            '1minute': ('minutes', '1'),
            '5minute': ('minutes', '5'),
            '15minute': ('minutes', '15'),
            '30minute': ('minutes', '30'),
            '1hour': ('hours', '1'),
            '1day': ('days', '1')
        }

    def _make_request(self, method: str, url: str, **kwargs) -> Dict:
        """
        Make an API request with error handling.
        
        Parameters:
        -----------
        method : str
            HTTP method (GET, POST, PUT, DELETE)
        url : str
            API endpoint URL
        **kwargs : dict
            Additional arguments for requests
        
        Returns:
        --------
        Dict
            API response
        """
        try:
            response = requests.request(method, url, headers=self.headers, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API Request Error: {str(e)}")
            if hasattr(e.response, 'json'):
                return e.response.json()
            return {"status": "error", "message": str(e)}

    def _get_instrument_key(self, symbol: str) -> str:
        """
        Get instrument key for a symbol using symbol_codes module.
        
        Parameters:
        -----------
        symbol : str
            Trading symbol
        
        Returns:
        --------
        str
            Instrument key
        """
        try:
            # First check if symbol exists and get its details
            check_symbol_details(symbol)
            # Get the instrument key
            instrument_key = get_instrument_key(symbol)
            # Format the instrument key according to Upstox API requirements
            if '|' in instrument_key:
                return instrument_key
            else:
                # If not in correct format, construct it
                return f"NSE_EQ|{instrument_key}"
        except Exception as e:
            logger.error(f"Error getting instrument key for {symbol}: {str(e)}")
            return None

    def _get_instrument_keys(self, symbols: List[str]) -> List[str]:
        """
        Convert symbols to instrument keys using symbol_codes module.
        
        Parameters:
        -----------
        symbols : List[str]
            List of trading symbols
        
        Returns:
        --------
        List[str]
            List of instrument keys
        """
        return [self._get_instrument_key(symbol) for symbol in symbols]

    def _process_candle_data(self, candles: List) -> pd.DataFrame:
        """
        Process candle data into a DataFrame.
        
        Parameters:
        -----------
        candles : List
            List of candle data
        
        Returns:
        --------
        pd.DataFrame
            Processed candle data
        """
        if not candles:
            logger.warning("No candle data found in the response")
            return pd.DataFrame()
        
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.sort_values('timestamp')

    def _process_market_quotes(self, quotes_data: Dict) -> pd.DataFrame:
        """
        Process market quotes data into a DataFrame.
        
        Parameters:
        -----------
        quotes_data : Dict
            Market quotes data from API
        
        Returns:
        --------
        pd.DataFrame
            Processed market quotes data
        """
        if not quotes_data or 'data' not in quotes_data:
            logger.warning("No market quotes data found in the response")
            return pd.DataFrame()
        
        # Extract quotes data
        quotes = quotes_data['data']
        
        # Create a list to store processed data
        processed_data = []
        
        # Process each quote
        for instrument_key, quote in quotes.items():
            try:
                # Extract symbol from instrument key
                # Format: NSE_EQ|ISIN_CODE or NSE_EQ:SYMBOL
                if '|' in instrument_key:
                    parts = instrument_key.split('|')
                    if len(parts) != 2:
                        logger.warning(f"Invalid instrument key format: {instrument_key}")
                        continue
                    exchange, isin = parts
                    # Find symbol from ISIN
                    symbol = None
                    for sym, isin_code in SYMBOL_TO_ISIN.items():
                        if isin_code == isin:
                            symbol = sym
                            break
                    if not symbol:
                        logger.warning(f"Symbol not found for ISIN: {isin}")
                        continue
                elif ':' in instrument_key:
                    parts = instrument_key.split(':')
                    if len(parts) != 2:
                        logger.warning(f"Invalid instrument key format: {instrument_key}")
                        continue
                    exchange, symbol = parts
                    # Find ISIN from symbol
                    isin = SYMBOL_TO_ISIN.get(symbol.upper())
                    if not isin:
                        logger.warning(f"ISIN not found for symbol: {symbol}")
                        continue
                else:
                    logger.warning(f"Invalid instrument key format: {instrument_key}")
                    continue
                
                # Convert last_trade_time from milliseconds to datetime
                last_trade_time = quote.get('last_trade_time')
                if last_trade_time:
                    try:
                        # Convert milliseconds to seconds and then to datetime
                        last_trade_time = pd.to_datetime(int(last_trade_time) / 1000, unit='s')
                    except (ValueError, TypeError, OverflowError) as e:
                        logger.warning(f"Error converting last_trade_time: {str(e)}")
                        last_trade_time = None
                
                # Create a dictionary with the quote data
                quote_dict = {
                    'symbol': symbol,
                    'exchange': exchange,
                    'isin': isin,
                    'last_price': quote.get('last_price', None),
                    'open': quote.get('ohlc', {}).get('open', None),
                    'high': quote.get('ohlc', {}).get('high', None),
                    'low': quote.get('ohlc', {}).get('low', None),
                    'close': quote.get('ohlc', {}).get('close', None),
                    'volume': quote.get('volume', None),
                    'change': quote.get('change', None),
                    'change_percent': quote.get('change_percent', None),
                    'last_trade_time': last_trade_time,
                    'oi': quote.get('oi', None),
                    'prev_close': quote.get('prev_close', None),
                    'prev_open': quote.get('prev_open', None),
                    'prev_high': quote.get('prev_high', None),
                    'prev_low': quote.get('prev_low', None),
                    'prev_volume': quote.get('prev_volume', None),
                    'prev_oi': quote.get('prev_oi', None)
                }
                processed_data.append(quote_dict)
                
            except Exception as e:
                logger.error(f"Error processing quote for {instrument_key}: {str(e)}")
                continue
        
        # Create DataFrame
        df = pd.DataFrame(processed_data)
        
        return df

    def get_intraday_data(self, symbol: str, timeframe: str = '1minute') -> pd.DataFrame:
        """
        Fetch intraday candle data for the current trading day.
        
        Parameters:
        -----------
        symbol : str
            Trading symbol
        timeframe : str
            Timeframe for the data (default: '1minute')
            Options: 
            - minutes: 1, 2, 3, ..., 300
            - hours: 1, 2, ..., 5
            - days: 1
        
        Returns:
        --------
        pd.DataFrame
            Intraday candle data with columns:
            - timestamp: Start time of the candle
            - open: Opening price
            - high: Highest price
            - low: Lowest price
            - close: Closing price
            - volume: Trading volume
            - oi: Open Interest
        """
        try:
            # Parse timeframe
            if 'minute' in timeframe:
                unit = 'minutes'
                interval = timeframe.replace('minute', '')
            elif 'hour' in timeframe:
                unit = 'hours'
                interval = timeframe.replace('hour', '')
            elif 'day' in timeframe:
                unit = 'days'
                interval = '1'
            else:
                raise ValueError(f"Invalid timeframe format: {timeframe}")
            
            # Validate interval based on unit
            if unit == 'minutes' and not (1 <= int(interval) <= 300):
                raise ValueError("For minutes, interval must be between 1 and 300")
            elif unit == 'hours' and not (1 <= int(interval) <= 5):
                raise ValueError("For hours, interval must be between 1 and 5")
            
            # Get instrument key
            instrument_key = self._get_instrument_key(symbol)
            
            # Make API request
            url = f'{self.base_url}/v3/historical-candle/intraday/{instrument_key}/{unit}/{interval}'
            logger.info(f"Fetching intraday data from URL: {url}")
            
            response = self._make_request('GET', url)
            
            if response.get('status') == 'error':
                error_msg = response.get('message', 'Unknown error')
                logger.error(f"API Error: {error_msg}")
                return pd.DataFrame()
            
            candles = response.get('data', {}).get('candles', [])
            if not candles:
                logger.warning(f"No intraday data found for {symbol}")
                return pd.DataFrame()
            
            return self._process_candle_data(candles)
            
        except Exception as e:
            logger.error(f"Error fetching intraday data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def get_historical_data(self, symbol: str, from_date: str, to_date: str, timeframe: str = '1minute') -> pd.DataFrame:
        """
        Get historical candle data for a symbol
        
        Args:
            symbol: Trading symbol
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            timeframe: Candle timeframe (1minute, 5minute, 15minute, 30minute, 1hour, 1day)
            
        Returns:
            pd.DataFrame: Historical candle data
        """
        try:
            # Validate and parse dates
            try:
                from_dt = datetime.strptime(from_date, '%Y-%m-%d')
                to_dt = datetime.strptime(to_date, '%Y-%m-%d')
            except ValueError:
                logger.error("Invalid date format. Use YYYY-MM-DD")
                return pd.DataFrame()
            
            # Check if dates are in the future
            current_date = datetime.now()
            if from_dt > current_date or to_dt > current_date:
                logger.error("Cannot fetch data for future dates")
                return pd.DataFrame()
            
            # Ensure from_date is before to_date
            if from_dt >= to_dt:
                logger.error("from_date must be before to_date")
                return pd.DataFrame()
            
            # Parse timeframe to determine unit and interval
            if 'minute' in timeframe:
                unit = 'minutes'
                interval = int(timeframe.replace('minute', ''))
                # For 1-15 minute intervals, limit to 1 month
                if interval <= 15:
                    max_days = 30
                else:
                    # For >15 minute intervals, limit to 1 quarter
                    max_days = 90
            elif 'hour' in timeframe:
                unit = 'hours'
                interval = int(timeframe.replace('hour', ''))
                max_days = 90  # 1 quarter
            elif 'day' in timeframe:
                unit = 'days'
                interval = 1
                max_days = 3650  # 1 decade
            elif 'week' in timeframe:
                unit = 'weeks'
                interval = 1
                max_days = None  # No limit
            elif 'month' in timeframe:
                unit = 'months'
                interval = 1
                max_days = None  # No limit
            else:
                logger.error(f"Invalid timeframe format: {timeframe}")
                return pd.DataFrame()
            
            # Check date range limits
            date_diff = (to_dt - from_dt).days
            if max_days and date_diff > max_days:
                logger.warning(f"Date range exceeds {max_days} days limit for {timeframe} timeframe. Adjusting start date.")
                from_dt = to_dt - timedelta(days=max_days)
                from_date = from_dt.strftime('%Y-%m-%d')
            
            # Get instrument key
            instrument_key = self._get_instrument_key(symbol)
            if not instrument_key:
                logger.error(f"Could not get instrument key for {symbol}")
                return pd.DataFrame()
            
            # Make API request
            url = f"{self.base_url}/v3/historical-candle/{instrument_key}/{unit}/{interval}/{to_date}/{from_date}"
            logger.info(f"Fetching historical data from: {url}")
            
            response = self._make_request('GET', url)
            
            if response.get('status') == 'error':
                error_msg = response.get('message', 'Unknown error')
                logger.error(f"API Error: {error_msg}")
                return pd.DataFrame()
            
            candles = response.get('data', {}).get('candles', [])
            if not candles:
                logger.warning(f"No historical data found for {symbol} from {from_date} to {to_date}")
                return pd.DataFrame()
            
            df = self._process_candle_data(candles)
            
            # Save to CSV
            try:
                os.makedirs('historical_data', exist_ok=True)
                filename = f"historical_data/{symbol}_{timeframe}_{from_date}_to_{to_date}.csv"
                df.to_csv(filename, index=False)
                logger.info(f"Historical data saved to {filename}")
            except Exception as e:
                logger.error(f"Error saving historical data: {str(e)}")
            
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            if hasattr(e.response, 'text'):
                logger.error(f"API response: {e.response.text}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            return pd.DataFrame()

    def get_full_market_quote(self, symbols: List[str]) -> pd.DataFrame:
        """
        Get full market quotes for multiple symbols.
        
        Parameters:
        -----------
        symbols : List[str]
            List of trading symbols
        
        Returns:
        --------
        pd.DataFrame
            Market quotes data with columns:
            - symbol
            - last_price
            - open
            - high
            - low
            - close
            - volume
            - change
            - change_percent
            - last_trade_time
            - oi
            - prev_close
            - prev_open
            - prev_high
            - prev_low
            - prev_volume
            - prev_oi
        """
        instrument_keys = self._get_instrument_keys(symbols)
        if not instrument_keys:
            return pd.DataFrame()
        
        url = f"{self.base_url}/v2/market-quote/quotes"
        params = {'instrument_key': ','.join(instrument_keys)}
        response = self._make_request('GET', url, params=params)
        
        return self._process_market_quotes(response)

    def get_ohlc_quotes(self, symbols: List[str], interval: str = '1d') -> pd.DataFrame:
        """
        Get OHLC quotes for multiple symbols.
        
        Parameters:
        -----------
        symbols : List[str]
            List of trading symbols
        interval : str
            Time interval for OHLC data
            Options: '1d' (1 day), 'I1' (1 minute), 'I30' (30 minutes)
        
        Returns:
        --------
        pd.DataFrame
            OHLC quotes with columns:
            - symbol: Trading symbol
            - live_ohlc: Current OHLC candle
            - prev_ohlc: Previous minute's OHLC candle
            - volume: Trading volume
            - ts: OHLC candle's start time
        """
        try:
            # Validate interval
            valid_intervals = ['1d', 'I1', 'I30']
            if interval not in valid_intervals:
                raise ValueError(f"Invalid interval. Choose from: {valid_intervals}")
            
            # Get instrument keys
            instrument_keys = self._get_instrument_keys(symbols)
            if not instrument_keys:
                return pd.DataFrame()
            
            # Check maximum limit
            if len(instrument_keys) > 500:
                logger.warning("Maximum 500 instrument keys allowed. Truncating list.")
                instrument_keys = instrument_keys[:500]
            
            # Make API request
            url = f"{self.base_url}/v3/market-quote/ohlc"
            params = {
                'instrument_key': ','.join(instrument_keys),
                'interval': interval
            }
            
            response = self._make_request('GET', url, params=params)
            
            if response.get('status') == 'error':
                error_msg = response.get('message', 'Unknown error')
                logger.error(f"API Error: {error_msg}")
                return pd.DataFrame()
            
            # Process response
            data = response.get('data', {})
            if not data:
                logger.warning("No OHLC data received")
                return pd.DataFrame()
            
            # Create DataFrame
            quotes_data = []
            for instrument_key, quote in data.items():
                try:
                    symbol = self._get_symbol_from_instrument_key(instrument_key)
                    if not symbol:
                        continue
                    
                    quote_dict = {
                        'symbol': symbol,
                        'live_ohlc': quote.get('live_ohlc', {}),
                        'prev_ohlc': quote.get('prev_ohlc', {}),
                        'volume': quote.get('volume'),
                        'ts': quote.get('ts')
                    }
                    quotes_data.append(quote_dict)
                except Exception as e:
                    logger.error(f"Error processing quote for {instrument_key}: {str(e)}")
                    continue
                
            return pd.DataFrame(quotes_data)
            
        except Exception as e:
            logger.error(f"Error fetching OHLC quotes: {str(e)}")
            return pd.DataFrame()

    def get_ltp_quotes(self, symbols: List[str]) -> pd.DataFrame:
        """
        Get Last Traded Price (LTP) quotes for multiple symbols.
        
        Parameters:
        -----------
        symbols : List[str]
            List of trading symbols
        
        Returns:
        --------
        pd.DataFrame
            LTP quotes with columns:
            - symbol: Trading symbol
            - ltp: Last traded price
            - volume: Trading volume
            - ts: Timestamp
        """
        try:
            # Get instrument keys
            instrument_keys = self._get_instrument_keys(symbols)
            if not instrument_keys:
                return pd.DataFrame()
            
            # Check maximum limit
            if len(instrument_keys) > 500:
                logger.warning("Maximum 500 instrument keys allowed. Truncating list.")
                instrument_keys = instrument_keys[:500]
            
            # Make API request
            url = f"{self.base_url}/v3/market-quote/ltp"
            params = {'instrument_key': ','.join(instrument_keys)}
            
            response = self._make_request('GET', url, params=params)
            
            if response.get('status') == 'error':
                error_msg = response.get('message', 'Unknown error')
                logger.error(f"API Error: {error_msg}")
                return pd.DataFrame()
            
            # Process response
            data = response.get('data', {})
            if not data:
                logger.warning("No LTP data received")
                return pd.DataFrame()
            
            # Create DataFrame
            quotes_data = []
            for instrument_key, quote in data.items():
                try:
                    symbol = self._get_symbol_from_instrument_key(instrument_key)
                    if not symbol:
                        continue
                    
                    quote_dict = {
                        'symbol': symbol,
                        'ltp': quote.get('ltp'),
                        'volume': quote.get('volume'),
                        'ts': quote.get('ts')
                    }
                    quotes_data.append(quote_dict)
                except Exception as e:
                    logger.error(f"Error processing quote for {instrument_key}: {str(e)}")
                    continue
                
            return pd.DataFrame(quotes_data)
            
        except Exception as e:
            logger.error(f"Error fetching LTP quotes: {str(e)}")
            return pd.DataFrame()

    def get_option_greeks(self, option_symbols: List[str]) -> pd.DataFrame:
        """
        Get option Greeks for multiple option symbols.
        
        Parameters:
        -----------
        option_symbols : List[str]
            List of option symbols (e.g., ['NIFTY50-25APR2024-22000-CE'])
            Format: UNDERLYING-EXPIRY-STRIKE-TYPE
            TYPE: CE for Call, PE for Put
        
        Returns:
        --------
        pd.DataFrame
            Option Greeks data with columns:
            - symbol: Option symbol
            - underlying: Underlying asset
            - expiry: Expiry date
            - strike: Strike price
            - option_type: Call or Put
            - delta: Delta value
            - gamma: Gamma value
            - theta: Theta value
            - vega: Vega value
            - rho: Rho value
            - implied_volatility: Implied volatility
        """
        try:
            # Process option symbols
            processed_data = []
            for symbol in option_symbols:
                try:
                    # Parse option symbol
                    parts = symbol.split('-')
                    if len(parts) != 4:
                        raise ValueError(f"Invalid option symbol format: {symbol}")
                    
                    underlying, expiry, strike, option_type = parts
                    
                    # Validate option type
                    if option_type not in ['CE', 'PE']:
                        raise ValueError(f"Invalid option type: {option_type}. Must be CE or PE")
                    
                    # Create instrument key
                    instrument_key = f'NSE_FO|{symbol}'
                    
                    # Make API request
                    url = f"{self.base_url}/v2/market-quote/option-greeks"
                    params = {'instrument_key': instrument_key}
                    response = self._make_request('GET', url, params=params)
                    
                    if response and 'data' in response and instrument_key in response['data']:
                        greeks_data = response['data'][instrument_key]
                        
                        # Create dictionary with option data
                        option_dict = {
                            'symbol': symbol,
                            'underlying': underlying,
                            'expiry': expiry,
                            'strike': float(strike),
                            'option_type': option_type,
                            'delta': greeks_data.get('delta'),
                            'gamma': greeks_data.get('gamma'),
                            'theta': greeks_data.get('theta'),
                            'vega': greeks_data.get('vega'),
                            'rho': greeks_data.get('rho'),
                            'implied_volatility': greeks_data.get('implied_volatility')
                        }
                        processed_data.append(option_dict)
                    else:
                        logger.warning(f"No Greeks data found for {symbol}")
                        
                except Exception as e:
                    logger.error(f"Error processing option symbol {symbol}: {str(e)}")
                    continue
            
            # Create DataFrame
            df = pd.DataFrame(processed_data)
            
            # Convert expiry to datetime
            if 'expiry' in df.columns:
                df['expiry'] = pd.to_datetime(df['expiry'], format='%d%b%Y')
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching option Greeks: {str(e)}")
            return pd.DataFrame()

    def get_market_depth(self, symbols: List[str]) -> Dict:
        """
        Get market depth (order book) for multiple symbols.
        
        Parameters:
        -----------
        symbols : List[str]
            List of trading symbols
        
        Returns:
        --------
        Dict
            Market depth
        """
        instrument_keys = self._get_instrument_keys(symbols)
        if not instrument_keys:
            return {"status": "error", "message": "No valid symbols provided"}
        
        url = f"{self.base_url}/v2/market-quote/depth-full"
        params = {'instrument_key': ','.join(instrument_keys)}
        return self._make_request('GET', url, params=params)

    def _get_symbol_from_instrument_key(self, instrument_key: str) -> Optional[str]:
        """
        Extract symbol from instrument key.
        
        Parameters:
        -----------
        instrument_key : str
            Instrument key in format 'NSE_EQ|ISIN' or 'NSE_EQ:SYMBOL'
        
        Returns:
        --------
        Optional[str]
            Trading symbol if found, None otherwise
        """
        try:
            if '|' in instrument_key:
                parts = instrument_key.split('|')
                if len(parts) != 2:
                    return None
                exchange, isin = parts
                # Find symbol from ISIN
                for sym, isin_code in SYMBOL_TO_ISIN.items():
                    if isin_code == isin:
                        return sym
            elif ':' in instrument_key:
                parts = instrument_key.split(':')
                if len(parts) != 2:
                    return None
                exchange, symbol = parts
                return symbol.upper()
            return None
        except Exception as e:
            logger.error(f"Error extracting symbol from instrument key: {str(e)}")
            return None

# Example usage
if __name__ == "__main__":
    # Initialize market data manager
    market_data = UpstoxMarketData()
    
    # Example: Get intraday data
    symbol = "TATAMOTORS"
    timeframe = "5minute"
    intraday_df = market_data.get_intraday_data(symbol, timeframe)
    if not intraday_df.empty:
        print("\nIntraday Data:")
        print(intraday_df.head())
        print(f"Total candles: {len(intraday_df)}")
    
    # Example: Get historical data
    from_date = "2024-04-01"
    to_date = "2024-04-02"
    historical_df = market_data.get_historical_data(symbol, from_date, to_date, timeframe)
    if not historical_df.empty:
        print("\nHistorical Data:")
        print(historical_df.head())
        print(f"Total candles: {len(historical_df)}")
    
    # Example: Get market quotes
    symbols = ["TATAMOTORS", "RELIANCE"]
    quotes_df = market_data.get_full_market_quote(symbols)
    if not quotes_df.empty:
        print("\nMarket Quotes:")
        print(quotes_df)
    
    # Example: Get option Greeks
    option_symbols = ["NIFTY50-25APR2024-22000-CE", "NIFTY50-25APR2024-22000-PE"]
    greeks_df = market_data.get_option_greeks(option_symbols)
    if not greeks_df.empty:
        print("\nOption Greeks:")
        print(greeks_df) 