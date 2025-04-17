import os
import logging
import pandas as pd
from modules.snowflake_connector import SnowflakeDWH
from config.settings import SNOWFLAKE_CONFIG

logger = logging.getLogger(__name__)

def run_query_to_pandas_file_example():
    """Example of running a query from a SQL file and retrieving results as a DataFrame."""
    try:
        # Initialize Snowflake connector
        snowflake = SnowflakeDWH()
        
        # Determine the path to the SQL file
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sql_file_path = os.path.join(base_dir, 'sql_queries', 'holiday_basic.sql')
        
        # Check if the file exists
        if not os.path.exists(sql_file_path):
            logger.error(f"SQL file not found: {sql_file_path}")
            return pd.DataFrame()
        
        # Define parameters for the SQL query
        query_params = {
            "{database}": SNOWFLAKE_CONFIG['DATABASE'],
            "{schema}": SNOWFLAKE_CONFIG['SCHEMA'],
            "{table}": SNOWFLAKE_CONFIG['HOLIDAY_TABLE'],
            "{min_date}": "'2025-01-01'"  # Note the quotes for SQL string literal
        }
        
        logger.info(f"Executing SQL file: {sql_file_path}")
        logger.debug(f"Using parameters: {query_params}")
        
        # Execute query and get results as DataFrame
        results_df = snowflake.query_to_pandas(
            filename=sql_file_path,
            **query_params
        )
        
        # Example of processing the results
        logger.info(f"Retrieved {len(results_df)} rows from Snowflake")
        
        # Example of filtering and analyzing the data
        irrenunciable_holidays = results_df[results_df['irrenunciable'] == True]
        logger.info(f"Found {len(irrenunciable_holidays)} non-waivable holidays")
        print(results_df)
        
        # Return the DataFrame for further processing
        return results_df
        
    except Exception as e:
        logger.error(f"Error in query example: {e}", exc_info=True)
        return pd.DataFrame()

if __name__ == "__main__":
    # Run the example
    run_query_to_pandas_file_example()