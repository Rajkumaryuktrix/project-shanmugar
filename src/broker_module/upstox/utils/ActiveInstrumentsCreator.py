"""
Module for creating and managing active instruments data.
"""

import json
import os
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ActiveInstrumentsCreator:
    def __init__(self, 
                 bse_file: str = "src/BrokerModule/instruments/BSE.json",
                 nse_file: str = "src/BrokerModule/instruments/NSE.json",
                 output_file: str = "src/BrokerModule/instruments/active_instruments.json"):
        """
        Initialize the ActiveInstrumentsCreator with paths to instrument files.
        
        Args:
            bse_file (str): Path to the BSE instruments JSON file
            nse_file (str): Path to the NSE instruments JSON file
            output_file (str): Path to save the combined instruments JSON file
        """
        self.bse_file = bse_file
        self.nse_file = nse_file
        self.output_file = output_file
        self.combined_instruments = {}
        
    def _load_json_file(self, file_path: str) -> List[Dict]:
        """
        Load data from a JSON file.
        
        Args:
            file_path (str): Path to the JSON file
            
        Returns:
            List[Dict]: List of instrument data
        """
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            return []
            
    def combine_instruments(self) -> None:
        """
        Combine BSE and NSE instrument data into a single dictionary with trading_symbol as key.
        """
        # Load BSE and NSE data
        bse_data = self._load_json_file(self.bse_file)
        nse_data = self._load_json_file(self.nse_file)
        
        # Process BSE data
        for instrument in bse_data:
            if 'trading_symbol' in instrument:
                self.combined_instruments[instrument['trading_symbol']] = instrument
                
        # Process NSE data
        for instrument in nse_data:
            if 'trading_symbol' in instrument:
                # If symbol already exists (from BSE), add exchange suffix
                if instrument['trading_symbol'] in self.combined_instruments:
                    symbol = f"{instrument['trading_symbol']}_NSE"
                else:
                    symbol = instrument['trading_symbol']
                self.combined_instruments[symbol] = instrument
                
        logger.info(f"Combined {len(bse_data)} BSE and {len(nse_data)} NSE instruments")
        
    def save_combined_instruments(self) -> None:
        """
        Save the combined instruments data to the output JSON file.
        """
        try:
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            with open(self.output_file, 'w') as f:
                json.dump(self.combined_instruments, f, indent=2)
            logger.info(f"Combined instruments saved to {self.output_file}")
        except Exception as e:
            logger.error(f"Error saving combined instruments: {str(e)}")

# Example usage
if __name__ == "__main__":
    creator = ActiveInstrumentsCreator()
    
    # Combine BSE and NSE instruments
    creator.combine_instruments()
    
    # Save the combined data
    creator.save_combined_instruments() 