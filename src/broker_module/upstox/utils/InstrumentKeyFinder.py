"""
Module for finding instrument keys based on company names.
"""

import json
import os
import logging
from difflib import get_close_matches
from typing import List, Dict, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InstrumentKeyFinder:
    def __init__(self, instruments_file: str = None):
        """
        Initialize the InstrumentKeyFinder with the path to the instruments JSON file.
        
        Args:
            instruments_file (str): Path to the active_instruments.json file
        """
        if instruments_file is None:
            # Get the absolute path to the instruments file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.instruments_file = os.path.join(current_dir, 'instruments', 'active_instruments.json')
        else:
            self.instruments_file = instruments_file
            
        self.instruments_data = self._load_instruments()
        
    def _load_instruments(self) -> Dict:
        """
        Load the instruments data from the JSON file.
        
        Returns:
            Dict: Dictionary containing instrument data
        """
        try:
            with open(self.instruments_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Instruments file not found: {self.instruments_file}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from file: {self.instruments_file}")
            raise
            
    def _find_exact_match(self, search_text: str) -> Optional[Tuple[str, str]]:
        """
        Find an exact match for the search text in the instrument keys.
        
        Args:
            search_text (str): Text to search for
            
        Returns:
            Optional[Tuple[str, str]]: Tuple of (key, instrument_key) if found, None otherwise
        """
        search_text = search_text.upper()
        for key, data in self.instruments_data.items():
            if search_text == key:
                return (key, data.get('instrument_key'))
        return None
        
    def _find_similar_matches(self, search_text: str, n: int = 3) -> List[Tuple[str, str]]:
        """
        Find similar matches for the search text in the instrument keys.
        
        Args:
            search_text (str): Text to search for
            n (int): Number of similar matches to return
            
        Returns:
            List[Tuple[str, str]]: List of tuples containing (key, instrument_key)
        """
        search_text = search_text.upper()
        keys = list(self.instruments_data.keys())
        similar_keys = get_close_matches(search_text, keys, n=n, cutoff=0.6)
        return [(key, self.instruments_data[key].get('instrument_key')) for key in similar_keys]
        
    def find_instrument_key(self, search_text: str) -> str:
        """
        Find the instrument key for the given search text.
        If no exact match is found, displays similar matches and asks for user input.
        
        Args:
            search_text (str): Text to search for
            
        Returns:
            str: Selected instrument key
        """
        # Try to find exact match first
        exact_match = self._find_exact_match(search_text)
        if exact_match:
            logger.info(f"Found exact match: {exact_match[0]}")
            return exact_match[1]
            
        # If no exact match, find similar matches
        similar_matches = self._find_similar_matches(search_text)
        if not similar_matches:
            logger.error(f"No matches found for: {search_text}")
            raise ValueError(f"No matches found for: {search_text}")
            
        # Display similar matches and get user input
        print("\nNo exact match found. Similar matches:")
        for i, (key, instrument_key) in enumerate(similar_matches, 1):
            print(f"{i}. {key}")
            
        while True:
            try:
                choice = int(input("\nEnter the number of your choice (or 0 to exit): "))
                if choice == 0:
                    raise ValueError("Search cancelled by user")
                if 1 <= choice <= len(similar_matches):
                    selected_key, selected_instrument_key = similar_matches[choice - 1]
                    logger.info(f"Selected instrument key: {selected_instrument_key}")
                    return selected_instrument_key
                print("Invalid choice. Please try again.")
            except ValueError as e:
                if str(e) == "Search cancelled by user":
                    raise
                print("Please enter a valid number.")

# Example usage
if __name__ == "__main__":
    finder = InstrumentKeyFinder()
    try:
        search_text = input("Enter company name to search: ")
        instrument_key = finder.find_instrument_key(search_text)
        print(f"\nInstrument Key: {instrument_key}")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
