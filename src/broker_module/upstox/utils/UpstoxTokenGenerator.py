"""
Module for generating Upstox API access tokens.
"""

import webbrowser
import requests
from dotenv import load_dotenv
import os
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UpstoxTokenGenerator:
    def __init__(self):
        """Initialize the UpstoxTokenGenerator."""
        self.auth_code = None
        self.server = None
        self.redirect_uri = "http://localhost:8000/callback"
        self.token_url = "https://api.upstox.com/v2/login/authorization/token"
        
        # Load environment variables
        load_dotenv(override=True)
        self.api_key = os.getenv('UPSTOX_API_KEY')
        self.api_secret = os.getenv('UPSTOX_API_SECRET')

    class CallbackHandler(BaseHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            self.token_generator = kwargs.pop('token_generator')
            super().__init__(*args, **kwargs)

        def do_GET(self):
            # Extract the code from the URL
            query_components = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            if 'code' in query_components:
                self.token_generator.auth_code = query_components['code'][0]
                # Send a response to the browser
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(b"Authorization successful! You can close this window and return to the terminal.")
            else:
                self.send_response(400)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(b"Authorization failed! No code received.")

    def _start_local_server(self):
        """Start a local server to handle the OAuth callback."""
        server = HTTPServer(('localhost', 8000), 
                          lambda *args, **kwargs: self.CallbackHandler(*args, token_generator=self, **kwargs))
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()
        return server

    def _verify_credentials(self):
        """Verify that API credentials are present in environment variables."""
        logger.info("\nEnvironment Variables Check:")
        logger.info(f"API Key exists: {'Yes' if self.api_key else 'No'}")
        logger.info(f"API Secret exists: {'Yes' if self.api_secret else 'No'}")
        
        if not self.api_key:
            logger.error("\nERROR: UPSTOX_API_KEY not found in .env file")
            logger.error("Please make sure your .env file contains:")
            logger.error("UPSTOX_API_KEY=your_api_key_here")
            return False
        
        if not self.api_secret:
            logger.error("\nERROR: UPSTOX_API_SECRET not found in .env file")
            logger.error("Please make sure your .env file contains:")
            logger.error("UPSTOX_API_SECRET=your_api_secret_here")
            return False
        
        return True

    def _get_authorization_code(self):
        """Get the authorization code through OAuth flow."""
        # Construct the authorization URL
        auth_params = {
            'response_type': 'code',
            'client_id': self.api_key,
            'redirect_uri': self.redirect_uri
        }
        auth_url = f"https://api.upstox.com/v2/login/authorization/dialog?{urllib.parse.urlencode(auth_params)}"
        
        logger.info("\nPlease follow these steps:")
        logger.info("1. Opening browser for Upstox login...")
        webbrowser.open(auth_url)
        
        logger.info("\n2. After logging in, you'll be redirected to a page with an authorization code")
        logger.info("3. Wait for the authorization code to be automatically captured...")
        
        # Wait for the authorization code
        max_wait_time = 300  # 5 minutes
        start_time = time.time()
        
        while not self.auth_code and (time.time() - start_time) < max_wait_time:
            time.sleep(1)
        
        if not self.auth_code:
            logger.error("\nTimeout: No authorization code received within 5 minutes")
            return False
        
        logger.info(f"\nAuthorization code received: {self.auth_code}")
        return True

    def _exchange_code_for_token(self):
        """Exchange the authorization code for an access token."""
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        data = {
            'code': self.auth_code,
            'client_id': self.api_key,
            'client_secret': self.api_secret,
            'redirect_uri': self.redirect_uri,
            'grant_type': 'authorization_code'
        }
        
        try:
            # Format the data as form-urlencoded string
            encoded_data = urllib.parse.urlencode(data)
            
            # Make the request
            response = requests.post(
                self.token_url,
                headers=headers,
                data=encoded_data
            )
            
            if response.status_code != 200:
                logger.error(f"Error Response: {response.text}")
                return None
                
            response.raise_for_status()
            token_data = response.json()
            
            logger.info("\nToken generated successfully!")
            logger.info("\nAdd these to your .env file:")
            logger.info(f"UPSTOX_ACCESS_TOKEN={token_data['access_token']}")
            
            return token_data['access_token']
            
        except requests.exceptions.RequestException as e:
            logger.error(f"\nError generating token: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response Status: {e.response.status_code}")
                logger.error(f"Response Headers: {dict(e.response.headers)}")
                try:
                    logger.error(f"Response Body: {e.response.json()}")
                except:
                    logger.error(f"Response Text: {e.response.text}")
            return None

    def generate_token(self):
        """
        Generate a new Upstox API access token.
        
        Returns:
            str: The generated access token, or None if generation failed
        """
        try:
            # Verify credentials
            if not self._verify_credentials():
                return None
            
            # Start local server
            self.server = self._start_local_server()
            
            # Get authorization code
            if not self._get_authorization_code():
                return None
            
            # Exchange code for token
            return self._exchange_code_for_token()
            
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return None
            
        finally:
            # Shutdown the local server
            if self.server:
                self.server.shutdown()

def main():
    """Main function to run the token generator."""
    generator = UpstoxTokenGenerator()
    generator.generate_token()

if __name__ == "__main__":
    main() 
    