# API Client Module
## Overview
This module provides a flexible framework for interacting with RESTful APIs. It includes a base client class that handles common HTTP operations and a specialized client for fetching holiday data.

## Classes
### BaseAPIClient
A foundation class for creating API clients with standardized request handling.
```python
BaseAPIClient(base_url: str, headers: dict = None)
```

#### Parameters:

- `base_url`: The base URL for the API (without endpoints)
- `headers`: Optional custom headers for API requests. Defaults to a standard User-Agent if not provided.

#### Methods:

- `make_request(endpoint: str, method: str = 'GET', params: dict = None, data: dict = None)`: Makes an HTTP request to the specified endpoint and returns the JSON response.

### HolidayAPIClient
A specialized client for fetching holiday data from the Boostr API.

```python
HolidayAPIClient(base_url: str = 'https://api.boostr.cl/holidays')
```

#### Parameters:
- `base_url`: The base URL for the holiday API. Defaults to 'https://api.boostr.cl/holidays'.

#### Methods:
- `get_holidays(year: int) -> Optional[pd.DataFrame]`: Fetches holidays for a specific year and returns the data as a pandas DataFrame.

## Usage Examples
### Basic Usage
```python
# Create a holiday API client
client = HolidayAPIClient()

# Fetch holidays for a specific year
holidays_df = client.get_holidays(2025)

# Process the data
if holidays_df is not None:
    print(f"Found {len(holidays_df)} holidays for 2025")
    print(holidays_df.head())
```

### Creating a Custom API Client
```python
# Create a custom API client for a different service
weather_client = BaseAPIClient(
    base_url="https://api.weatherservice.com/v1",
    headers={
        "Authorization": "Bearer YOUR_API_KEY",
        "Content-Type": "application/json"
    }
)

# Make a request to get weather data
weather_data = weather_client.make_request(
    endpoint="forecast",
    params={"city": "Santiago", "days": 7}
)

# Process the response
if weather_data:
    print(f"Weather forecast: {weather_data}")
```


### Error Handling
The module includes comprehensive error handling:

- HTTP errors are caught and logged.
- Data processing errors are handled gracefully.
- Missing or invalid responses return None instead of raising exceptions.
- All errors are logged using the standard Python logging module.

### Dependencies
- `requests`: For making HTTP requests.
- `pandas`: For data manipulation and DataFrame creation.
- `logging`: For error and information logging.
- `typing`: For type hints.

### Extending the Module
To create a new API client for a different service:

1. Inherit from `BaseAPIClient`.
2. Override the constructor if needed.
3. Create service-specific methods that use `make_request()`.
Example:

```python
class ProductAPIClient(BaseAPIClient):
    """Client for fetching product data."""
    
    def __init__(self, api_key: str):
        super().__init__(
            base_url="https://api.products.com/v2",
            headers={"Authorization": f"Bearer {api_key}"}
        )
    
    def get_product(self, product_id: str) -> dict:
        """Fetch a specific product by ID."""
        return self.make_request(f"products/{product_id}")
    
    def search_products(self, query: str, limit: int = 10) -> list:
        """Search for products matching the query."""
        return self.make_request(
            endpoint="search",
            params={"q": query, "limit": limit}
        )
```

### Best Practices
- Always check if the return value is `None` before processing.
- Use the logging module to track API interactions.
- Consider implementing rate limiting for APIs with restrictions.
- Add timeout parameters for production use.