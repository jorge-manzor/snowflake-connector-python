import logging
from modules.api_client import HolidayAPIClient
from modules.get_holidays_chile import HolidayDataProcessor
from modules.snowflake_connector import SnowflakeDWH
from config.settings import SNOWFLAKE_CONFIG, PROCESSING_CONFIG

logger = logging.getLogger(__name__)

def process_year_range(
    start_year: int = PROCESSING_CONFIG['DEFAULT_START_YEAR'], 
    end_year: int = PROCESSING_CONFIG['DEFAULT_END_YEAR'],
    database: str = SNOWFLAKE_CONFIG['DATABASE'],
    schema: str = SNOWFLAKE_CONFIG['SCHEMA'],
    table: str = SNOWFLAKE_CONFIG['HOLIDAY_TABLE']
):
    """Process a range of years and load holiday data to Snowflake."""
    api_client = HolidayAPIClient()
    processor = HolidayDataProcessor()
    snowflake = SnowflakeDWH()
    
    years = list(range(start_year, end_year + 1))
    
    for year in years:
        logger.info(f"Processing holiday data for year: {year}")
        
        # Get holidays from API
        holidays_df = api_client.get_holidays(year)
        
        # Process the data
        holidays_df = processor.add_custom_holidays(holidays_df, year)
        holidays_df = processor.filter_holidays(holidays_df)
        holidays_df = processor.add_metadata(holidays_df)
        
        # Merge to Snowflake
        snowflake.merge_dataframe(
            dataframe=holidays_df,
            database_name=database,
            schema_name=schema,
            target_table=table,
            key_columns=["fecha_feriado"],
            update_columns=["nombre", "tipo", "irrenunciable", 'additional', 'src_updated_at']
        )
