"""
Module for handling symbol to ISIN code conversions and instrument management.
"""

import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, Optional, List
import logging

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# File paths
CACHE_FILE = 'instrument_cache.json'
CACHE_EXPIRY_DAYS = 1  # Update cache every day

def load_instruments_from_json() -> list:
    """
    Load instruments from NSE.json and BSE.json files.
    
    Returns:
        list: List of instrument data
    """
    all_instruments = []
    try:
        # Load from NSE.json
        nse_file = os.path.join(os.path.dirname(__file__), 'NSE.json')
        logger.info(f"Loading instruments from {nse_file}")
        with open(nse_file, 'r') as f:
            nse_data = json.load(f)
            logger.info(f"Loaded {len(nse_data)} instruments from NSE.json")
            all_instruments.extend(nse_data)
            
        # Load from BSE.json
        bse_file = os.path.join(os.path.dirname(__file__), 'BSE.json')
        logger.info(f"Loading instruments from {bse_file}")
        with open(bse_file, 'r') as f:
            bse_data = json.load(f)
            logger.info(f"Loaded {len(bse_data)} instruments from BSE.json")
            all_instruments.extend(bse_data)
            
        logger.info(f"Total instruments loaded: {len(all_instruments)}")
        return all_instruments
    except Exception as e:
        logger.error(f"Error loading instruments from JSON: {str(e)}")
        raise

def load_cached_instruments() -> Optional[Dict]:
    """
    Load cached instrument data if available and not expired.
    
    Returns:
        Optional[Dict]: Cached instrument data or None if expired/not found
    """
    if not os.path.exists(CACHE_FILE):
        return None
        
    try:
        with open(CACHE_FILE, 'r') as f:
            cache_data = json.load(f)
            
        # Check if cache is expired
        last_updated = datetime.fromisoformat(cache_data['last_updated'])
        if (datetime.now() - last_updated).days >= CACHE_EXPIRY_DAYS:
            return None
            
        return cache_data['instruments']
    except Exception as e:
        logger.error(f"Error loading cached instruments: {str(e)}")
        return None

def save_instruments_to_cache(instruments: Dict):
    """
    Save instrument data to cache file.
    
    Args:
        instruments (Dict): Instrument data to cache
    """
    try:
        cache_data = {
            'last_updated': datetime.now().isoformat(),
            'instruments': instruments
        }
        
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache_data, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving instruments to cache: {str(e)}")

def check_symbol_details(symbol: str) -> None:
    """
    Check and print detailed information about a specific symbol.
    
    Args:
        symbol (str): Trading symbol to check
    """
    symbol = symbol.upper()
    logger.info(f"\nChecking details for symbol: {symbol}")
    
    # Check in NSE and BSE files
    try:
        # Check NSE.json
        nse_file = os.path.join(os.path.dirname(__file__), 'NSE.json')
        with open(nse_file, 'r') as f:
            nse_instruments = json.load(f)
            for instrument in nse_instruments:
                if instrument.get('trading_symbol', '').upper() == symbol:
                    segment = instrument.get('segment', '')
                    instrument_type = instrument.get('instrument_type', '')
                    instrument_key = instrument.get('instrument_key', '')
                    logger.info(f"Found {symbol} in NSE:")
                    logger.info(f"  Segment: {segment}")
                    logger.info(f"  Instrument Type: {instrument_type}")
                    logger.info(f"  Instrument Key: {instrument_key}")
                    if segment != 'NSE_EQ' or instrument_type != 'EQ':
                        logger.warning(f"Symbol {symbol} is not an equity instrument")
                    return
        
        # Check BSE.json
        bse_file = os.path.join(os.path.dirname(__file__), 'BSE.json')
        with open(bse_file, 'r') as f:
            bse_instruments = json.load(f)
            for instrument in bse_instruments:
                if instrument.get('trading_symbol', '').upper() == symbol:
                    segment = instrument.get('segment', '')
                    instrument_type = instrument.get('instrument_type', '')
                    instrument_key = instrument.get('instrument_key', '')
                    logger.info(f"Found {symbol} in BSE:")
                    logger.info(f"  Segment: {segment}")
                    logger.info(f"  Instrument Type: {instrument_type}")
                    logger.info(f"  Instrument Key: {instrument_key}")
                    if segment != 'BSE_EQ' or instrument_type != 'EQ':
                        logger.warning(f"Symbol {symbol} is not an equity instrument")
                    return
                    
        logger.info(f"Symbol {symbol} not found in NSE or BSE files")
                
    except Exception as e:
        logger.error(f"Error checking symbol details: {str(e)}")

def update_instrument_mapping() -> Dict:
    """
    Update the instrument mapping using NSE.json and BSE.json files.
    Focuses only on equity instruments.
    
    Returns:
        Dict: Updated symbol to instrument key mapping
    """
    logger.info("Forcing refresh of instrument data")
    SYMBOL_TO_ISIN = {}
    
    try:
        # Load from NSE.json
        nse_file = os.path.join(os.path.dirname(__file__), 'NSE.json')
        logger.info(f"Loading instruments from {nse_file}")
        with open(nse_file, 'r') as f:
            nse_instruments = json.load(f)
            logger.info(f"Loaded {len(nse_instruments)} instruments from NSE.json")
            
            # Process NSE instruments
            for instrument in nse_instruments:
                symbol = instrument.get('trading_symbol')
                if not symbol:
                    continue
                    
                symbol = symbol.upper()
                segment = instrument.get('segment', '')
                instrument_type = instrument.get('instrument_type', '')
                instrument_key = instrument.get('instrument_key', '')
                
                # Log the first few instruments for debugging
                if len(SYMBOL_TO_ISIN) < 5:
                    logger.info(f"Sample NSE instrument: {symbol}")
                    logger.info(f"  Segment: {segment}")
                    logger.info(f"  Type: {instrument_type}")
                    logger.info(f"  Key: {instrument_key}")
                
                # Check if it's an equity instrument
                if (segment == 'NSE_EQ' and 
                    instrument_type == 'EQ' and 
                    instrument_key):
                    # Store instrument key as the primary identifier
                    SYMBOL_TO_ISIN[symbol] = {
                        'instrument_key': instrument_key,
                        'exchange': 'NSE'
                    }
                    logger.debug(f"Added NSE {symbol} with key {instrument_key}")
                else:
                    logger.debug(f"Skipping NSE {symbol}: segment={segment}, type={instrument_type}")
        
        # Load from BSE.json
        bse_file = os.path.join(os.path.dirname(__file__), 'BSE.json')
        logger.info(f"Loading instruments from {bse_file}")
        with open(bse_file, 'r') as f:
            bse_instruments = json.load(f)
            logger.info(f"Loaded {len(bse_instruments)} instruments from BSE.json")
            
            # Process BSE instruments
            for instrument in bse_instruments:
                symbol = instrument.get('trading_symbol')
                if not symbol:
                    continue
                    
                symbol = symbol.upper()
                segment = instrument.get('segment', '')
                instrument_type = instrument.get('instrument_type', '')
                instrument_key = instrument.get('instrument_key', '')
                
                # Log the first few instruments for debugging
                if len(SYMBOL_TO_ISIN) < 5:
                    logger.info(f"Sample BSE instrument: {symbol}")
                    logger.info(f"  Segment: {segment}")
                    logger.info(f"  Type: {instrument_type}")
                    logger.info(f"  Key: {instrument_key}")
                
                # Check if it's an equity instrument
                if (segment == 'BSE_EQ' and 
                    instrument_type == 'EQ' and 
                    instrument_key):
                    # Store instrument key as the primary identifier
                    SYMBOL_TO_ISIN[symbol] = {
                        'instrument_key': instrument_key,
                        'exchange': 'BSE'
                    }
                    logger.debug(f"Added BSE {symbol} with key {instrument_key}")
                else:
                    logger.debug(f"Skipping BSE {symbol}: segment={segment}, type={instrument_type}")
                    
    except Exception as e:
        logger.error(f"Error processing instruments: {str(e)}")
        if not SYMBOL_TO_ISIN:
            logger.warning("No instruments were processed. Using empty mapping.")
    
    # Save to cache
    save_instruments_to_cache(SYMBOL_TO_ISIN)
    
    # Log the total number of instruments processed
    logger.info(f"Total equity instruments processed: {len(SYMBOL_TO_ISIN)}")
    
    # Log some sample symbols for verification
    sample_symbols = list(SYMBOL_TO_ISIN.keys())[:5]
    logger.info(f"Sample equity symbols in mapping: {sample_symbols}")
    
    return SYMBOL_TO_ISIN

# Initialize the mapping
try:
    logger.info("Initializing instrument mapping")
    SYMBOL_TO_ISIN = update_instrument_mapping()
except Exception as e:
    logger.error(f"Failed to initialize instrument mapping: {str(e)}")
    SYMBOL_TO_ISIN = {}

def get_isin_code(symbol: str) -> str:
    """
    Get ISIN code for a given equity symbol.
    
    Args:
        symbol (str): Trading symbol
        
    Returns:
        str: ISIN code
        
    Raises:
        ValueError: If symbol is not found or is not an equity instrument
    """
    symbol = symbol.upper()
    if symbol not in SYMBOL_TO_ISIN:
        # Check symbol details before raising error
        check_symbol_details(symbol)
        raise ValueError(f"ISIN code not found for symbol: {symbol}")
    
    isin = SYMBOL_TO_ISIN[symbol]['isin']
    if not isinstance(isin, str):
        raise ValueError(f"Invalid ISIN format for symbol {symbol}")
    return isin

def get_instrument_key(symbol: str) -> str:
    """
    Get instrument key for a given equity symbol.
    
    Args:
        symbol (str): Trading symbol
        
    Returns:
        str: Instrument key
        
    Raises:
        ValueError: If symbol is not found or is not an equity instrument
    """
    symbol = symbol.upper()
    if symbol not in SYMBOL_TO_ISIN:
        # Check symbol details before raising error
        check_symbol_details(symbol)
        raise ValueError(f"Instrument key not found for symbol: {symbol}")
    return SYMBOL_TO_ISIN[symbol]['instrument_key']

def get_exchange(symbol: str) -> str:
    """
    Get exchange for a given equity symbol.
    
    Args:
        symbol (str): Trading symbol
        
    Returns:
        str: Exchange (NSE or BSE)
        
    Raises:
        ValueError: If symbol is not found or is not an equity instrument
    """
    symbol = symbol.upper()
    if symbol not in SYMBOL_TO_ISIN:
        # Check symbol details before raising error
        check_symbol_details(symbol)
        raise ValueError(f"Exchange not found for symbol: {symbol}")
    return SYMBOL_TO_ISIN[symbol]['exchange']

def get_symbol(isin_or_key: str) -> str:
    """
    Get symbol for a given ISIN code or instrument key.
    
    Args:
        isin_or_key (str): ISIN code or instrument key
        
    Returns:
        str: Trading symbol
        
    Raises:
        ValueError: If ISIN/key is not found
    """
    for symbol, details in SYMBOL_TO_ISIN.items():
        if details['isin'] == isin_or_key or details['instrument_key'] == isin_or_key:
            return symbol
            
    # Try to find in the instruments list
    try:
        instruments = load_instruments_from_json()
        for instrument in instruments:
            if (instrument.get('isin') == isin_or_key or 
                instrument.get('instrument_key') == isin_or_key):
                return instrument.get('trading_symbol', '')
    except Exception as e:
        logger.error(f"Error searching for ISIN/key {isin_or_key}: {str(e)}")
        
    raise ValueError(f"Symbol not found for ISIN/key: {isin_or_key}")

def get_available_symbols() -> List[str]:
    """
    Get list of all available trading symbols.
    
    Returns:
        List[str]: List of available trading symbols
    """
    try:
        instruments = load_instruments_from_json()
        return sorted([instrument.get('trading_symbol', '') for instrument in instruments 
                      if instrument.get('trading_symbol')])
    except Exception as e:
        logger.error(f"Error getting available symbols: {str(e)}")
        return list(SYMBOL_TO_ISIN.keys())

def add_symbol_mapping(symbol: str, isin_or_key: str):
    """
    Add a new symbol mapping.
    
    Args:
        symbol (str): Trading symbol
        isin_or_key (str): ISIN code or instrument key
    """
    symbol = symbol.upper()
    # Try to determine if it's an ISIN or instrument key
    if isin_or_key.startswith('INE'):
        SYMBOL_TO_ISIN[symbol] = {
            'isin': isin_or_key,
            'instrument_key': f"NSE_EQ|{isin_or_key}",  # Default to NSE format
            'exchange': 'NSE'
        }
    else:
        SYMBOL_TO_ISIN[symbol] = {
            'isin': isin_or_key,  # Use as ISIN
            'instrument_key': isin_or_key,
            'exchange': 'NSE'  # Default to NSE
        }
    save_instruments_to_cache(SYMBOL_TO_ISIN)

def remove_symbol_mapping(symbol: str):
    """
    Remove a symbol from the mapping.
    
    Args:
        symbol (str): Trading symbol to remove
    """
    symbol = symbol.upper()
    if symbol in SYMBOL_TO_ISIN:
        del SYMBOL_TO_ISIN[symbol]
        save_instruments_to_cache(SYMBOL_TO_ISIN)

def get_option_symbol(underlying: str, strike_price: float, expiry: str, option_type: str) -> str:
    """
    Generate option symbol for a given underlying, strike price, expiry and option type.
    
    Args:
        underlying (str): Underlying symbol
        strike_price (float): Strike price
        expiry (str): Expiry date in YYYY-MM-DD format
        option_type (str): Option type (CE/PE)
        
    Returns:
        str: Option symbol
    """
    underlying = underlying.upper()
    option_type = option_type.upper()
    
    if option_type not in ['CE', 'PE']:
        raise ValueError("Option type must be 'CE' or 'PE'")
        
    # Format: SYMBOL + EXPIRY + STRIKE + OPTION_TYPE
    expiry_date = expiry.replace('-', '')
    strike = int(strike_price)
    return f"{underlying}{expiry_date}{strike}{option_type}"

def get_future_symbol(underlying: str, expiry: str) -> str:
    """
    Generate future symbol for a given underlying and expiry.
    
    Args:
        underlying (str): Underlying symbol
        expiry (str): Expiry date in YYYY-MM-DD format
        
    Returns:
        str: Future symbol
    """
    underlying = underlying.upper()
    expiry_date = expiry.replace('-', '')
    return f"{underlying}{expiry_date}FUT" 

def refresh_instrument_mapping():
    """
    Force refresh the instrument mapping from JSON file.
    """
    global SYMBOL_TO_ISIN
    SYMBOL_TO_ISIN = update_instrument_mapping() 

def find_similar_symbols(input_symbol: str, all_symbols: List[str], top_n: int = 3) -> List[str]:
    """
    Find similar symbols based on input using Levenshtein distance.
    
    Args:
        input_symbol (str): Input symbol to match
        all_symbols (List[str]): List of all available symbols
        top_n (int): Number of top matches to return
        
    Returns:
        List[str]: List of top N similar symbols
    """
    from difflib import get_close_matches
    return get_close_matches(input_symbol.upper(), all_symbols, n=top_n, cutoff=0.6) 