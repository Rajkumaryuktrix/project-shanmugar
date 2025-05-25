import requests
import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from symbol_codes import get_isin_code
import logging

logger = logging.getLogger(__name__)

class UpstoxOrderManager:
    """
    A class to handle all order-related operations using Upstox API.
    Includes functionality for placing, modifying, and canceling orders,
    as well as retrieving order and trade information.
    """
    
    def __init__(self, environment="live"):
        """
        Initialize the order manager with API credentials.
        
        Parameters:
        -----------
        environment : str
            API environment to use ('live' or 'sandbox')
        """
        load_dotenv()
        self.environment = environment.lower()
        
        # Set base URL based on environment
        if self.environment == "sandbox":
            self.base_url = "https://api-sandbox.upstox.com"
            self.access_token = os.getenv('UPSTOX_SANDBOX_ACCESS_TOKEN')
        else:
            self.base_url = "https://api.upstox.com"
            self.access_token = os.getenv('UPSTOX_ACCESS_TOKEN')
            
        if not self.access_token:
            raise ValueError(f"UPSTOX_{'SANDBOX_' if self.environment == 'sandbox' else ''}ACCESS_TOKEN not found in environment variables")
        
        self.headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.access_token}'
        }
        
        logger.info(f"Initialized UpstoxOrderManager in {self.environment.upper()} environment")

    def _make_request(self, method, url, **kwargs):
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
        dict
            API response
        """
        try:
            response = requests.request(method, url, headers=self.headers, **kwargs)
            response.raise_for_status()  # Raise exception for 4XX/5XX responses
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API Request Error: {str(e)}")
            if hasattr(e.response, 'json'):
                return e.response.json()
            return {"status": "error", "message": str(e)}

    def place_order(self, symbol, quantity, order_type="MARKET", transaction_type="BUY", 
                   product="D", validity="DAY", price=0, trigger_price=0, tag="", 
                   disclosed_quantity=0, is_amo=False, slice=False):
        """
        Place a single order using Upstox API V3.
        
        Parameters:
        -----------
        symbol : str
            Trading symbol (e.g., 'TATAMOTORS')
        quantity : int
            Order quantity
        order_type : str
            Type of order (MARKET, LIMIT, SL, SL-M)
        transaction_type : str
            BUY or SELL
        product : str
            Product type (D for delivery, I for intraday)
        validity : str
            Order validity (DAY, IOC)
        price : float
            Order price (0 for market orders)
        trigger_price : float
            Trigger price for SL orders
        tag : str
            Order tag for identification
        disclosed_quantity : int
            Disclosed quantity for iceberg orders
        is_amo : bool
            After Market Order flag
        slice : bool
            Enable order slicing for large quantities
        
        Returns:
        --------
        dict
            Order response with order IDs
        """
        isin_code = get_isin_code(symbol)
        instrument_key = f'NSE_EQ|{isin_code}'
        
        payload = {
            "quantity": quantity,
            "product": product,
            "validity": validity,
            "price": price,
            "tag": tag,
            "instrument_token": instrument_key,
            "order_type": order_type,
            "transaction_type": transaction_type,
            "disclosed_quantity": disclosed_quantity,
            "trigger_price": trigger_price,
            "is_amo": is_amo,
            "slice": slice
        }
        
        url = f"{self.base_url}/v3/order/place"
        return self._make_request('POST', url, json=payload)

    def place_multi_order(self, orders):
        """
        Place multiple orders simultaneously using Upstox API.
        
        Parameters:
        -----------
        orders : list
            List of order dictionaries, each containing:
            - correlation_id (required)
            - quantity
            - product
            - validity
            - price
            - tag
            - instrument_token
            - order_type
            - transaction_type
            - disclosed_quantity
            - trigger_price
            - is_amo
            - slice
        
        Returns:
        --------
        dict
            Response with order IDs for each order
        """
        url = f"{self.base_url}/v3/order/place-multi"
        return self._make_request('POST', url, json=orders)

    def modify_order(self, order_id, quantity=None, price=None, 
                    trigger_price=None, disclosed_quantity=None):
        """
        Modify an existing order using Upstox API V3.
        
        Parameters:
        -----------
        order_id : str
            ID of the order to modify
        quantity : int, optional
            New quantity
        price : float, optional
            New price
        trigger_price : float, optional
            New trigger price
        disclosed_quantity : int, optional
            New disclosed quantity
        
        Returns:
        --------
        dict
            Modified order details
        """
        payload = {}
        if quantity is not None:
            payload["quantity"] = quantity
        if price is not None:
            payload["price"] = price
        if trigger_price is not None:
            payload["trigger_price"] = trigger_price
        if disclosed_quantity is not None:
            payload["disclosed_quantity"] = disclosed_quantity
            
        url = f"{self.base_url}/v3/order/modify/{order_id}"
        return self._make_request('PUT', url, json=payload)

    def cancel_order(self, order_id):
        """
        Cancel an existing order using Upstox API V3.
        
        Parameters:
        -----------
        order_id : str
            ID of the order to cancel
        
        Returns:
        --------
        dict
            Cancellation response
        """
        url = f"{self.base_url}/v3/order/cancel/{order_id}"
        return self._make_request('DELETE', url)

    def cancel_multi_order(self, order_ids):
        """
        Cancel multiple orders simultaneously.
        
        Parameters:
        -----------
        order_ids : list
            List of order IDs to cancel
        
        Returns:
        --------
        dict
            Cancellation response for all orders
        """
        url = f"{self.base_url}/v3/order/cancel-multi"
        payload = {"order_ids": order_ids}
        return self._make_request('DELETE', url, json=payload)

    def exit_all_positions(self):
        """
        Exit all open positions.
        
        Returns:
        --------
        dict
            Response with exit order details
        """
        url = f"{self.base_url}/v3/order/exit-all-positions"
        return self._make_request('POST', url)

    def get_order_details(self, order_id):
        """
        Get details of a specific order.
        
        Parameters:
        -----------
        order_id : str
            ID of the order to fetch
        
        Returns:
        --------
        dict
            Order details
        """
        url = f"{self.base_url}/v3/order/{order_id}"
        return self._make_request('GET', url)

    def get_order_history(self, order_id):
        """
        Get history of a specific order.
        
        Parameters:
        -----------
        order_id : str
            ID of the order
        
        Returns:
        --------
        dict
            Order history
        """
        url = f"{self.base_url}/v3/order/history/{order_id}"
        return self._make_request('GET', url)

    def get_order_book(self):
        """
        Get all orders in the order book.
        
        Returns:
        --------
        dict
            List of all orders
        """
        try:
            # Different endpoints for live and sandbox
            if self.environment == "sandbox":
                url = f"{self.base_url}/v3/order/order-book"  # Sandbox endpoint
            else:
                url = f"{self.base_url}/v3/order/orders"  # Live endpoint
                
            response = self._make_request('GET', url)
            
            # Log the response for debugging
            logger.info(f"Order book response: {response}")
            
            if response.get('status') == 'error':
                logger.error(f"Error getting order book: {response.get('errors', [])}")
                return response
                
            return response
            
        except Exception as e:
            logger.error(f"Exception in get_order_book: {str(e)}")
            return {"status": "error", "message": str(e)}

    def get_trade_history(self, from_date=None, to_date=None):
        """
        Get trade history for a date range.
        
        Parameters:
        -----------
        from_date : str, optional
            Start date in YYYY-MM-DD format
        to_date : str, optional
            End date in YYYY-MM-DD format
        
        Returns:
        --------
        dict
            Trade history
        """
        url = f"{self.base_url}/v3/trade/history"
        params = {}
        if from_date:
            params["from_date"] = from_date
        if to_date:
            params["to_date"] = to_date
            
        return self._make_request('GET', url, params=params)

    def get_trades_by_order(self, order_id):
        """
        Get all trades for a specific order.
        
        Parameters:
        -----------
        order_id : str
            ID of the order
        
        Returns:
        --------
        dict
            List of trades for the order
        """
        url = f"{self.base_url}/v3/order/trades/{order_id}"
        return self._make_request('GET', url)

    def get_historical_trades(self, from_date=None, to_date=None):
        """
        Get historical trades for a date range.
        
        Parameters:
        -----------
        from_date : str, optional
            Start date in YYYY-MM-DD format
        to_date : str, optional
            End date in YYYY-MM-DD format
        
        Returns:
        --------
        dict
            Historical trades
        """
        url = f"{self.base_url}/v3/trade/history"
        params = {}
        if from_date:
            params["from_date"] = from_date
        if to_date:
            params["to_date"] = to_date
            
        return self._make_request('GET', url, params=params)

# Example usage
if __name__ == "__main__":
    # Get environment from user input
    environment = input("Enter environment (live/sandbox): ").lower()
    while environment not in ["live", "sandbox"]:
        print("Invalid environment. Please enter 'live' or 'sandbox'")
        environment = input("Enter environment (live/sandbox): ").lower()
    
    # Initialize order manager
    order_manager = UpstoxOrderManager(environment=environment)
    
    # Example: Place a market order
    order_response = order_manager.place_order(
        symbol="TATAMOTORS",
        quantity=1,
        order_type="MARKET",
        transaction_type="BUY"
    )
    print("Order Response:", order_response)
    
    # Example: Get order book
    order_book = order_manager.get_order_book()
    print("\nOrder Book:", order_book) 