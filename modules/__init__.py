"""
Core modules for Snowflake connector functionality.
"""

from .snowflake_connector import SnowflakeDWH
from .api_client import HolidayAPIClient, BaseAPIClient
from .get_holidays_chile import DataProcessor, HolidayDataProcessor

# Expose key classes at the modules level
__all__ = [
    'SnowflakeDWH'
    'HolidayAPIClient',
    'BaseAPIClient',
    'DataProcessor',
    'HolidayDataProcessor'
]