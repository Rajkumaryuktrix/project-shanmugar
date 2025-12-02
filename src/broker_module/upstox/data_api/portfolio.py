import requests
import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import logging
from typing import Dict, List, Optional, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UpstoxPortfolio:
    """
    A class to handle portfolio operations using Upstox API.
    Includes functionality for:
    - Getting positions
    - Getting MTF positions
    - Converting positions
    - Getting holdings
    - Getting report metadata
    - Getting P&L reports
    - Getting trade charges
    - Getting brokerage
    - Getting margin
    """
    
    def __init__(self):
        """Initialize the portfolio manager with API credentials."""
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

    def _process_positions(self, positions_data: Dict) -> pd.DataFrame:
        """
        Process positions data into a DataFrame.
        
        Parameters:
        -----------
        positions_data : Dict
            Positions data from API
        
        Returns:
        --------
        pd.DataFrame
            Processed positions data
        """
        if not positions_data or 'data' not in positions_data:
            logger.warning("No positions data found in the response")
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(positions_data['data'])
        
        # Convert numeric columns
        numeric_columns = [
            'multiplier', 'value', 'pnl', 'average_price', 'buy_value',
            'day_buy_value', 'day_buy_price', 'overnight_buy_amount',
            'day_sell_value', 'day_sell_price', 'overnight_sell_amount',
            'last_price', 'unrealised', 'realised', 'sell_value',
            'close_price', 'buy_price', 'sell_price'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert integer columns
        integer_columns = [
            'overnight_quantity', 'overnight_buy_quantity', 'day_buy_quantity',
            'overnight_sell_quantity', 'day_sell_quantity', 'quantity'
        ]
        
        for col in integer_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
        
        return df

    def get_positions(self) -> pd.DataFrame:
        """
        Get current positions for the user.
        
        Returns:
        --------
        pd.DataFrame
            Current positions with columns:
            - exchange
            - multiplier
            - value
            - pnl
            - product
            - instrument_token
            - average_price
            - buy_value
            - overnight_quantity
            - day_buy_value
            - day_buy_price
            - overnight_buy_amount
            - overnight_buy_quantity
            - day_buy_quantity
            - day_sell_value
            - day_sell_price
            - overnight_sell_amount
            - overnight_sell_quantity
            - day_sell_quantity
            - quantity
            - last_price
            - unrealised
            - realised
            - sell_value
            - trading_symbol
            - close_price
            - buy_price
            - sell_price
        """
        url = f"{self.base_url}/v2/portfolio/positions"
        response = self._make_request('GET', url)
        return self._process_positions(response)

    def get_mtf_positions(self) -> pd.DataFrame:
        """
        Get current Margin Trade Funding (MTF) positions for the user.
        
        Returns:
        --------
        pd.DataFrame
            MTF positions with the same columns as get_positions()
        """
        url = f"{self.base_url}/v2/portfolio/mtf-positions"
        response = self._make_request('GET', url)
        return self._process_positions(response)

    def convert_position(self, instrument_token: str, from_product: str, to_product: str, 
                        quantity: int) -> Dict:
        """
        Convert a position from one product type to another.
        
        Parameters:
        -----------
        instrument_token : str
            Instrument token of the position to convert
        from_product : str
            Current product type (I, D, CO)
        to_product : str
            Target product type (I, D, CO)
        quantity : int
            Quantity to convert
        
        Returns:
        --------
        Dict
            Conversion response
        """
        url = f"{self.base_url}/v2/portfolio/convert-position"
        payload = {
            "instrument_token": instrument_token,
            "from_product": from_product,
            "to_product": to_product,
            "quantity": quantity
        }
        return self._make_request('POST', url, json=payload)

    def get_holdings(self) -> pd.DataFrame:
        """
        Get current holdings for the user.
        
        Returns:
        --------
        pd.DataFrame
            Holdings data with columns:
            - exchange
            - trading_symbol
            - instrument_token
            - quantity
            - average_price
            - last_price
            - pnl
            - product
            - collateral_type
            - collateral_qty
            - haircut
            - collateral_value
            - isin
            - t1_quantity
            - realised_quantity
            - used_quantity
            - authorised_quantity
            - authorised_date
            - opening_quantity
            - used_quantity
            - used_quantity_date
            - used_quantity_price
            - used_quantity_value
            - used_quantity_pnl
            - used_quantity_realised
            - used_quantity_unrealised
            - used_quantity_net
            - used_quantity_net_pnl
            - used_quantity_net_realised
            - used_quantity_net_unrealised
        """
        url = f"{self.base_url}/v2/portfolio/holdings"
        response = self._make_request('GET', url)
        
        if not response or 'data' not in response:
            logger.warning("No holdings data found in the response")
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(response['data'])
        
        # Convert numeric columns
        numeric_columns = [
            'quantity', 'average_price', 'last_price', 'pnl', 'collateral_qty',
            'haircut', 'collateral_value', 't1_quantity', 'realised_quantity',
            'used_quantity', 'authorised_quantity', 'used_quantity_price',
            'used_quantity_value', 'used_quantity_pnl', 'used_quantity_realised',
            'used_quantity_unrealised', 'used_quantity_net', 'used_quantity_net_pnl',
            'used_quantity_net_realised', 'used_quantity_net_unrealised'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert date columns
        date_columns = ['authorised_date', 'used_quantity_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        return df

    def get_report_metadata(self, segment: str, financial_year: str, 
                          from_date: Optional[str] = None, to_date: Optional[str] = None) -> Dict:
        """
        Get metadata for P&L reports.
        
        Parameters:
        -----------
        segment : str
            Segment for which data is requested (EQ, FO, COM, CD)
        financial_year : str
            Financial year in format 'YYYY' (e.g., '2122' for 2021-2022)
        from_date : str, optional
            Start date in dd-mm-yyyy format
        to_date : str, optional
            End date in dd-mm-yyyy format
        
        Returns:
        --------
        Dict
            Report metadata with:
            - trades_count: Total number of trades
            - page_size_limit: Maximum page size
        """
        url = f"{self.base_url}/v2/trade/profit-loss/report-metadata"
        params = {
            'segment': segment,
            'financial_year': financial_year
        }
        
        if from_date:
            params['from_date'] = from_date
        if to_date:
            params['to_date'] = to_date
            
        return self._make_request('GET', url, params=params)

    def get_profit_loss_report(self, segment: str, financial_year: str, page_number: int,
                             page_size: int, from_date: Optional[str] = None, 
                             to_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get profit and loss report.
        
        Parameters:
        -----------
        segment : str
            Segment for which data is requested (EQ, FO, COM, CD)
        financial_year : str
            Financial year in format 'YYYY' (e.g., '2122' for 2021-2022)
        page_number : int
            Page number (starting from 1)
        page_size : int
            Number of records per page
        from_date : str, optional
            Start date in dd-mm-yyyy format
        to_date : str, optional
            End date in dd-mm-yyyy format
        
        Returns:
        --------
        pd.DataFrame
            P&L report data with columns:
            - quantity
            - isin
            - scrip_name
            - trade_type
            - buy_date
            - buy_average
            - sell_date
            - sell_average
            - buy_amount
            - sell_amount
        """
        url = f"{self.base_url}/v2/trade/profit-loss/report"
        params = {
            'segment': segment,
            'financial_year': financial_year,
            'page_number': page_number,
            'page_size': page_size
        }
        
        if from_date:
            params['from_date'] = from_date
        if to_date:
            params['to_date'] = to_date
            
        response = self._make_request('GET', url, params=params)
        
        if not response or 'data' not in response:
            logger.warning("No P&L report data found in the response")
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(response['data'])
        
        # Convert numeric columns
        numeric_columns = ['quantity', 'buy_average', 'sell_average', 'buy_amount', 'sell_amount']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert date columns
        date_columns = ['buy_date', 'sell_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], format='%d-%m-%Y')
        
        return df

    def get_trade_charges(self, segment: str, financial_year: str,
                         from_date: Optional[str] = None, to_date: Optional[str] = None) -> Dict:
        """
        Get trade charges for a period.
        
        Parameters:
        -----------
        segment : str
            Segment for which data is requested (EQ, FO, COM, CD)
        financial_year : str
            Financial year in format 'YYYY' (e.g., '2122' for 2021-2022)
        from_date : str, optional
            Start date in dd-mm-yyyy format
        to_date : str, optional
            End date in dd-mm-yyyy format
        
        Returns:
        --------
        Dict
            Trade charges data with:
            - total_charges
            - charges_breakdown
                - taxes
                    - gst
                    - stt
                    - stamp_duty
                - charges
                    - transaction
                    - clearing
                    - ipft
                    - others
                    - sebi_turnover
                    - demat_transaction
        """
        url = f"{self.base_url}/v2/trade/charges"
        params = {
            'segment': segment,
            'financial_year': financial_year
        }
        
        if from_date:
            params['from_date'] = from_date
        if to_date:
            params['to_date'] = to_date
            
        return self._make_request('GET', url, params=params)

    def get_brokerage(self, segment: str, financial_year: str,
                     from_date: Optional[str] = None, to_date: Optional[str] = None) -> Dict:
        """
        Get brokerage charges for a period.
        
        Parameters:
        -----------
        segment : str
            Segment for which data is requested (EQ, FO, COM, CD)
        financial_year : str
            Financial year in format 'YYYY' (e.g., '2122' for 2021-2022)
        from_date : str, optional
            Start date in dd-mm-yyyy format
        to_date : str, optional
            End date in dd-mm-yyyy format
        
        Returns:
        --------
        Dict
            Brokerage data with:
            - total_brokerage
            - brokerage_breakdown
                - equity
                - fno
                - commodity
                - currency
        """
        url = f"{self.base_url}/v2/trade/brokerage"
        params = {
            'segment': segment,
            'financial_year': financial_year
        }
        
        if from_date:
            params['from_date'] = from_date
        if to_date:
            params['to_date'] = to_date
            
        return self._make_request('GET', url, params=params)

    def get_margin(self, segment: str) -> Dict:
        """
        Get margin information for a segment.
        
        Parameters:
        -----------
        segment : str
            Segment for which data is requested (EQ, FO, COM, CD)
        
        Returns:
        --------
        Dict
            Margin data with:
            - available_margin
            - used_margin
            - opening_balance
            - closing_balance
            - margin_breakdown
                - equity
                - fno
                - commodity
                - currency
        """
        url = f"{self.base_url}/v2/margin"
        params = {'segment': segment}
        return self._make_request('GET', url, params=params)

# Example usage
if __name__ == "__main__":
    # Initialize portfolio manager
    portfolio = UpstoxPortfolio()
    
    # Get current positions
    positions_df = portfolio.get_positions()
    if not positions_df.empty:
        print("\nCurrent Positions:")
        print(positions_df)
    
    # Get MTF positions
    mtf_positions_df = portfolio.get_mtf_positions()
    if not mtf_positions_df.empty:
        print("\nMTF Positions:")
        print(mtf_positions_df)
    
    # Get holdings
    holdings_df = portfolio.get_holdings()
    if not holdings_df.empty:
        print("\nHoldings:")
        print(holdings_df)
    
    # Get report metadata
    metadata = portfolio.get_report_metadata("EQ", "2324")
    print("\nReport Metadata:", metadata)
    
    # Get P&L report
    pnl_df = portfolio.get_profit_loss_report("EQ", "2324", 1, 10)
    if not pnl_df.empty:
        print("\nP&L Report:")
        print(pnl_df)
    
    # Get trade charges
    charges = portfolio.get_trade_charges("EQ", "2324")
    print("\nTrade Charges:", charges)
    
    # Get brokerage
    brokerage = portfolio.get_brokerage("EQ", "2324")
    print("\nBrokerage:", brokerage)
    
    # Get margin
    margin = portfolio.get_margin("EQ")
    print("\nMargin:", margin) 