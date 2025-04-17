import requests
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class BaseAPIClient:
    """Base class for API clients."""
    
    def __init__(self, base_url: str, headers: dict = None):
        self.base_url = base_url
        self.headers = headers or {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
        }
    
    def make_request(self, endpoint: str, method: str = 'GET', params: dict = None, data: dict = None):
        """Make a request to the API."""
        url = f"{self.base_url}/{endpoint}"
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=self.headers,
                params=params,
                json=data
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request error: {e}")
            return None


class HolidayAPIClient(BaseAPIClient):
    """Client for fetching holiday data from external API."""
    
    def __init__(self, base_url: str = 'https://api.boostr.cl/holidays'):
        super().__init__(base_url)
    
    def get_holidays(self, year: int) -> Optional[pd.DataFrame]:
        """Fetch holidays for a specific year and return as DataFrame."""
        try:
            endpoint = f"{year}.json"
            logger.info(f"Fetching holidays for year {year}")
            
            holiday_data = self.make_request(endpoint)
            
            if not holiday_data or 'data' not in holiday_data:
                logger.warning(f"No holiday data found for year {year}")
                return None
                
            df = pd.DataFrame(holiday_data['data'])
            
            # Rename columns to standardized format
            column_mapping = {
                'date': 'fecha_feriado', 
                'title': 'nombre', 
                'type': 'tipo', 
                'inalienable': 'irrenunciable', 
                'extra': 'additional'
            }
            
            df = df.rename(columns=column_mapping)
            return df
            
        except (KeyError, ValueError) as e:
            logger.error(f"Data processing error: {e}")
            return None